import torch
import argparse
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, logging
from collections import Counter
from h_model import HierarchicalRoBERTa

# =========================
# ARGS PARSER
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        type=str,
                        default="data/kaggle_filtered.csv")
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

    return parser.parse_args()

# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    logging.set_verbosity_error()
    logging.disable_progress_bar()

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

    texts = df["self_text"].tolist()

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

    # statistics trackers
    total_texts = len(texts)
    total_foundations_predicted = 0

    # track polarity counts globally
    global_pol_counter = Counter()

    # track per-foundation polarity counts
    foundation_pol_counter = {
        f: Counter() for f in FOUNDATIONS
}

    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]

        found_preds, pol_preds = predict_batch(batch)

        for f_pred, p_pred in zip(found_preds, pol_preds):
            foundations_in_text = 0  # count per sample

            for idx, is_present in enumerate(f_pred):
                if is_present:
                    foundations_in_text += 1

                    found_label = FOUNDATIONS[idx]
                    pol_label = POLARITY[p_pred[idx].item()]

                    found_counter[found_label] += 1
                    pol_counter[f"{found_label}.{pol_label}"] += 1

                    # NEW: track per-foundation polarity
                    foundation_pol_counter[found_label][pol_label] += 1

                    # NEW: track global polarity
                    global_pol_counter[pol_label] += 1

            # NEW: accumulate total foundations predicted
            total_foundations_predicted += foundations_in_text

    
    # OUTPUT RESULTS
    print("\n=== FOUNDATION FREQUENCIES ===")
    for k,v in found_counter.items():
        print(f"{k}: {v}")

    print("\n=== FOUNDATION + POLARITY FREQUENCIES ===")
    for k,v in pol_counter.items():
        print(f"{k}: {v}")

    total_found = sum(found_counter.values())

    print("\n=== FOUNDATION DISTRIBUTION (%) ===")
    for k, v in found_counter.items():
        print(f"{k}: {v/total_found:.4f}")

    print("\n=== NORMALIZED POLARITY PER FOUNDATION ===")
    for foundation, pol_counts in foundation_pol_counter.items():
        total = sum(pol_counts.values())
        if total == 0:
            continue

        print(f"\n{foundation}:")
        for pol in POLARITY:
            val = pol_counts[pol] / total if total > 0 else 0
            print(f"  {pol}: {val:.4f}")

    print("\n=== GLOBAL POLARITY DISTRIBUTION (%) ===")
    total_global_pol = sum(global_pol_counter.values())

    for pol in POLARITY:
        val = global_pol_counter[pol] / total_global_pol if total_global_pol > 0 else 0
        print(f"{pol}: {val:.4f}")

    

    avg_foundations = total_foundations_predicted / total_texts if total_texts > 0 else 0
    print("\n=== AVERAGE FOUNDATIONS PER TEXT ===")
    print(f"{avg_foundations:.4f}")

if __name__ == "__main__":
    main()