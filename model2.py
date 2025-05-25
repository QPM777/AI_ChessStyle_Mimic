import modal
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset as HFDataset
from tqdm import tqdm
import torch

# Initialize Modal app and environment
stub = modal.App("flan-t5-train")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers>=4.36.0",
        "datasets",
        "sentencepiece",
        "pandas",
        "tqdm",
        "transformers[torch]"
    )
    .add_local_file("df_with_style.csv", remote_path="/root/df_with_style.csv")
)

# Custom PyTorch Dataset for move-by-move chess generation
class LiChessSequentialDataset(Dataset):
    def __init__(self, csv_file, max_context_moves=6):
        self.df = pd.read_csv(csv_file)
        self.examples = []
        self.max_context_moves = max_context_moves
        self.build_examples()

    # Parses PGN-like move string into raw moves (e.g., 1.e4 e5 -> [e4, e5])
    def parse_moves(self, move_str):
        tokens = move_str.strip().split()
        moves = [move for i, move in enumerate(tokens) if (i % 3) != 0]
        return moves

    # Builds encoder-decoder input/output pairs for T5
    def build_examples(self):
        for _, game in self.df.iterrows():
            base_desc = (
                f"In a {game['event']} on Lichess.org, {game['white']} played against {game['black']} "
                f"with the {game['opening']} opening. Ratings were {game['white_elo']} (White) vs {game['black_elo']} (Black). "
                f"The result was {game['result']} by {game['termination']}."
            )
            white_style = game.get("white_style", "Unknown")
            black_style = game.get("black_style", "Unknown")
            #TODO only white style and remove base descr 
            white_desc = f"Perspective: White. {base_desc} Style: {white_style}."
            black_desc = f"Perspective: Black. {base_desc} Style: {black_style}."

            full_moves = self.parse_moves(game["moves"])
            #TODO  add input description   and full_moves in output   
            for i in range(2, len(full_moves)):  # One move at a time
                context = ' '.join(full_moves[:i])
                target = full_moves[i]
                if not target:
                    continue
                # Alternate perspectives
                if i % 2 == 0:
                    self.examples.append({
                        "input": f"{white_desc} Moves so far: {context}",
                        "output": target,
                    })
                else:
                    self.examples.append({
                        "input": f"{black_desc} Moves so far: {context}",
                        "output": target,
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Converts dataset to dict format for Hugging Face
def prepare_data(dataset):
    return {"input": [ex["input"] for ex in dataset], "output": [ex["output"] for ex in dataset]}

# Modal-deployed training and inference function
@stub.function(image=image, timeout=10000, gpu="any")
def train_and_test(description_to_test=None):
    dataset = LiChessSequentialDataset("/root/df_with_style.csv")
    data_dict = prepare_data(dataset)
    hf_dataset = HFDataset.from_dict(data_dict)

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    def tokenize(batch):
        return tokenizer(
            batch["input"],
            text_target=batch["output"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized = hf_dataset.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = Seq2SeqTrainingArguments(
        output_dir="/root/output",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="/root/logs",
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        predict_with_generate=True,
        fp16=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Inference step
    if description_to_test:
        inputs = tokenizer(description_to_test, return_tensors="pt", max_length=512, truncation=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(
            inputs['input_ids'],
            max_length=20,
            num_beams=5,
            early_stopping=True
        )

        predicted_move = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return predicted_move

    return "Training complete."

# Local entrypoint for generating a chess game continuation step-by-step
@stub.local_entrypoint()
def main():
    # Empty list,  
    moves_played = ["1.d4", "d5"]
    description_base_white = (
        "Perspective: White. In a Bullet game on Lichess.org, pink_overdoze played against fil77 "
        "with the Queen's Pawn opening. Ratings were 1750 (White) vs 1350 (Black). Style: Unknown."
    )
    description_base_black = (
        "Perspective: Black. In a Bullet game on Lichess.org, pink_overdoze played against fil77 "
        "with the Queen's Pawn opening. Ratings were 1750 (White) vs 1350 (Black). Style: Unknown."
    )

    for step in range(10):  # Generate 10 moves (5 full turns)
        desc = description_base_white if step % 2 == 0 else description_base_black
        context = ' '.join(moves_played)
        full_input = f"{desc} Moves so far: {context}"
        predicted = train_and_test.remote(full_input)
        print(f"Step {step + 1} ({'White' if step % 2 == 0 else 'Black'}) prediction: {predicted}")
        moves_played.append(predicted)

    print("\nFinal continuation:")
    print(" ".join(moves_played))
#set modal perimeters in the code to increase the timer.
