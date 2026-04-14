import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torch.optim import AdamW
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from h_dataset import HierarchicalDataset
from h_model import HierarchicalRoBERTa
from h_trainer import train_epoch, evaluate

FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]

# =========================
# ARG PARSER
# =========================
'''
Parses CL args.
--data_dir: input directory
--output_dir: output directory for the saved model
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

    parser.add_argument("--model_dir", type=str,
                        default="models")
    
    parser.add_argument("--tokenizer_dir", type=str,
                        default="tokenizer")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lambda_weight", type=float, default=1.0)

    return parser.parse_args()
    

# =========================
# MAIN
# =========================
def main():
    '''
    Main method
    - loads the datasets, tokenizer, optimizer
    - frames CSVs as HierarchicalDataset(s)
    - uses data loaders to pass HDs in batches to training and evaluation functions
    - trains model for specified number of epochs
    '''
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(os.path.join(args.data_dir, "multi_label.csv"))

    label_matrix = df[FOUNDATIONS].values

    # structure the dataset with numerical labels
    for col in FOUNDATIONS:
        df[col] = df[col].astype(int)
    for col in [f"{f}_pol" for f in FOUNDATIONS]:
        df[col] = df[col].astype(int)

    # k-fold function
    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=args.max_len,
        return_tensors="pt"
    )

    fold_results = []

    # metadata for model saving
    best_f1 = 0
    best_model_state = None
    best_fold = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, label_matrix)):

        print(f"\n==============================")
        print(f"FOLD {fold+1}")
        print(f"==============================")

        # prepare the data into dataloaders
        train_dataset = HierarchicalDataset(df, encodings, train_idx)
        val_dataset = HierarchicalDataset(df, encodings, val_idx)

        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True)

        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=2,
                                pin_memory=True)

        # IMPORTANT: reset model per fold
        model = HierarchicalRoBERTa(lambda_weight=args.lambda_weight)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            train_loss, epoch_time = train_epoch(model, train_loader, optimizer, device, epoch)
            found_macro, found_micro, pol_macro_set, pol_micro_set, mean_pol_macro, mean_pol_micro  = evaluate(model, val_loader, device)

            print(f"\nEpoch {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Foundation Macro F1: {found_macro:.4f}")
            print(f"Foundation Micro F1: {found_micro:.4f}")
            print(f"Mean Polarity Macro F1: {mean_pol_macro:.4f}")
            print(f"Mean Polarity Micro F1: {mean_pol_micro:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")

            if ((found_macro + mean_pol_macro) / 2)> best_f1:
                best_f1 = mean_pol_macro
                best_model_state = model.state_dict()
                best_fold = fold

        fold_results.append((found_macro, mean_pol_macro))

    foundation_scores = [x[0] for x in fold_results]
    polarity_scores = [x[1] for x in fold_results]

    print(f"\n==============================")
    print(f"Summary of results")
    print(f"==============================")

    print("Foundation Macro F1 per fold: ", foundation_scores)
    print("Mean Foundation Macro F1: ", np.mean(foundation_scores))

    print("Mean Polarity Macro F1 per fold: ", polarity_scores)
    print("Mean Polarity Macro F1 overall: ", np.mean(polarity_scores))
    print("\n")
    
    os.makedirs(args.tokenizer_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    

    tokenizer.save_pretrained(args.tokenizer_dir)

    torch.save({
        "model_state_dict": best_model_state,
        "best_fold": best_fold,
        "best_f1": best_f1,
        "lambda_weight": args.lambda_weight,
        "max_len": args.max_len,
        "foundations": FOUNDATIONS
    }, os.path.join(args.output_dir, f"h_model_best.pt"))
    print(f"\nBest model saved from fold {best_fold} with F1={best_f1:.4f}.")


if __name__ == "__main__":
    main()