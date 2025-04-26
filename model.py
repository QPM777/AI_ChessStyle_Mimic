import modal
import torch
from datasets import Dataset as HFDataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


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
    .add_local_file("lichess_preprocessed.csv", remote_path="/root/lichess_preprocessed.csv")
)


class LiChessDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def parse_moves(self, move_str):
        tokens = move_str.strip().split()
        moves = [move for i, move in enumerate(tokens) if (i % 3) != 0]
        white_moves = moves[::2]
        black_moves = moves[1::2]
        return white_moves, black_moves, moves

    def __getitem__(self, idx):
        game = self.df.iloc[idx]
        base_description = (
            f"In a {game['event']} on Lichess, {game['white']} played against {game['black']} "
            f"with the {game['opening']} opening. Ratings were {game['white_elo']} (White) vs {game['black_elo']} (Black). "
            f"The result was {game['result']} by {game['termination']}."
        )

        white_style = game['white_style'] if 'white_style' in game and pd.notnull(game['white_style']) else "Unknown"
        black_style = game['black_style'] if 'black_style' in game and pd.notnull(game['black_style']) else "Unknown"

        white_encoder_input = f"{base_description} This is from White's perspective. Style: {white_style}"
        black_encoder_input = f"{base_description} This is from Black's perspective. Style: {black_style}"

        white_moves, black_moves, full_moves = self.parse_moves(game['moves'])
        decoder_input = ' '.join(full_moves)

        return {
            "white_encoder_input": white_encoder_input,
            "black_encoder_input": black_encoder_input,
            "decoder_input": decoder_input,
            "white_moves": ' '.join(white_moves),
            "black_moves": ' '.join(black_moves),
        }

def prepare_data(dataset):
    data = {
        "input": [],
        "output": [],
    }
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        data["input"].append(item["white_encoder_input"])  
        data["output"].append(item["decoder_input"])
    return data

@stub.function(image=image, timeout=1200, gpu="any")
def train_and_test(description_to_test=None):

    dataset = LiChessDataset("/root/lichess_preprocessed.csv")
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
        num_train_epochs=5,
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

 
    if description_to_test:

        inputs = tokenizer(description_to_test, return_tensors="pt", max_length=512, truncation=True)
        

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)


        inputs = {key: value.to(device) for key, value in inputs.items()}

       
        outputs = model.generate(
            inputs['input_ids'],  
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

        predicted_moves = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return predicted_moves
    
    return "End of training"

@stub.local_entrypoint()
def main():
    description = "In a Bullet game on Lichess.org, pink_overdoze played against fil77 with the Queen's Pawn opening. Ratings were 1750 (White) vs 1350 (Black). This is from White's perspective. Style: Unknown"
    
    result = train_and_test.remote(description)
    print(f"Description: {description}")
    print(f"Predicted moves: {result}")
