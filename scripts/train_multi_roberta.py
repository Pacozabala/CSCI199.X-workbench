import os
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torch.optim import AdamW

from nn_dataset import HierarchicalDataset
from nn_model import HierarchicalRoBERTa
from nn_trainer import train_epoch, evaluate

FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]

# =========================
# ARG PARSER
# =========================
'''
Parses CL args.
--data_dir: input directory
--output_dir: output directory
--epochs: number of training epochs
--batch_size: number of data entries processed at once.
--lr: learning rate.
--max_len: maximum length of input document.
--lambda_weight: lambda weight passed to the model.
'''
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        default="data/hierarchical_dataset")

    parser.add_argument("--output_dir", type=str,
                        default="outputs")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lambda_weight", type=float, default=1.0)

    return parser.parse_args()
    

# =========================
# MAIN
# =========================
'''
Main method
- loads the datasets, tokenizer, optimizer
- frames CSVs as HierarchicalDataset(s)
- uses data loaders to pass HDs in batches to training and evaluation functions
- trains model for specified number of epochs
'''
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))

    for col in FOUNDATIONS:
        train_df[col] = train_df[col].astype(int)
        val_df[col] = val_df[col].astype(int)
    for col in [f"{f}_pol" for f in FOUNDATIONS]:
        train_df[col] = train_df[col].astype(int)
        val_df[col] = val_df[col].astype(int)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = HierarchicalDataset(train_df, tokenizer, args.max_len)
    val_dataset = HierarchicalDataset(val_df, tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size)

    model = HierarchicalRoBERTa(lambda_weight=args.lambda_weight)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, epoch_time = train_epoch(model, train_loader, optimizer, device, epoch)
        foundation_f1, polarity_f1_set, mean_polarity_f1 = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Foundation Macro F1: {foundation_f1:.4f}")
        for f in range(5):
            print(f"{FOUNDATIONS[f]} Polarity Macro F1 (masked): {polarity_f1_set[f]:.4f}")
        print(f"Mean Polarity Macro F1 (masked): {mean_polarity_f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "hierarchical_model.pt"))

    print("\nModel saved.")


if __name__ == "__main__":
    main()