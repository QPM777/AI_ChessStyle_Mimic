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

from peft import get_peft_model, LoraConfig, TaskType

# Initialize Modal app and environment
stub = modal.App("flan-t5-lora-train")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers>=4.36.0",
        "datasets",
        "sentencepiece",
        "pandas",
        "tqdm",
        "transformers[torch]",
        "peft"
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

    def parse_moves(self, move_str):
        tokens = move_str.strip().split()
        moves = [move for i, move in enumerate(tokens) if (i % 3) != 0]
        return moves

    def build_examples(self):
        for _, game in self.df.iterrows():
            white_style = game.get("white_style", "Unknown")
            black_style = game.get("black_style", "Unknown")
            white_desc = f"Perspective: White. Style: {white_style}."
            black_desc = f"Perspective: Black. Style: {black_style}."

            full_moves = self.parse_moves(game["moves"])
            for i in range(2, len(full_moves)):
                context = ' '.join(full_moves[:i])
                target = full_moves[i]
                if not target:
                    continue
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
@stub.function(image=image, timeout=21600, gpu="any")  # 6 hours timeout
def train_and_test(description_to_test=None):
    dataset = LiChessSequentialDataset("/root/df_with_style.csv")
    data_dict = prepare_data(dataset)
    hf_dataset = HFDataset.from_dict(data_dict)

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    base_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],  # T5 attention modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

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
    moves_played = ["1.d4", "d5"]
    description_base_white = (
        "Perspective: White. Style: Unknown. Moves so far: 1.d4 d5"
    )
    description_base_black = (
        "Perspective: Black. Style: Unknown. Moves so far: 1.d4 d5"
    )

    for step in range(10):  # Generate 10 moves (5 full turns)
        desc = description_base_white if step % 2 == 0 else description_base_black
        context = ' '.join(moves_played)
        full_input = f"{desc[:-len('1.d4 d5')]}{context}"  # inject current context
        predicted = train_and_test.remote(full_input)
        print(f"Step {step + 1} ({'White' if step % 2 == 0 else 'Black'}) prediction: {predicted}")
        moves_played.append(predicted)

    print("\nFinal continuation:")
    print(" ".join(moves_played))
