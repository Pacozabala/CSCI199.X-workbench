import torch
import argparse
import pandas as pd

from transformers import AutoTokenizer
from collections import Counter
from h_model import HierarchicalRoBERTa

# =========================
# ARGS PARSER
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        type=str,
                        default="data/kaggle_dataset.csv") # TODO: adjust csv name later
    parser.add_argument("--model_path",
                        type=str,
                        default="models/h_model_best.pt")
    parser.add_argument("--tokenizer",
                        type=str,
                        default="tokenizer/")
    parser.add_argument("--output_path",
                        type=str,
                        default="outputs/")
    parser.add_argument("--max_len",
                        type=int,
                        default=128)
    parser.add_argument("--batch_size",
                        type=int,
                        default=32)

    return parser

# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    # CONFIG
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]
    POLARITY = ["virtue", "vice", "neutral"]

    # LOAD MODEL + TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = HierarchicalRoBERTa()
    checkpoint = torch.load(args.model_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # LOAD DATASET (KAGGLE CSV)
    df = pd.read_csv(args.data_path)

    texts = df["text"].tolist() # TODO: fix column name

    # INFERENCE FUNCTION
    def predict_batch(text_batch):
        encoding = tokenizer(
            text_batch,
            truncation=True,
            padding=True,
            max_length=args.max_len,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            _, found_logits, pol_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            found_probs = torch.sigmoid(found_logits)
            found_preds = (found_probs > 0.5).int()

            pol_preds = torch.argmax(pol_logits, dim=-1)

        return found_preds.cpu(), pol_preds.cpu()
    
    # RUN INFERENCE
    found_counter = Counter()
    pol_counter = Counter()

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]