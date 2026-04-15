import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, logging
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
--model_dir: output directory for the saved model
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

    logging.set_verbosity_error()
    logging.disable_progress_bar()

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

    # metadata for model saving
    best_pol_f1 = -1
    best_model_state = None
    best_fold = -1

    # arrays for training logs
    epoch_logs = []
    fold_logs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, label_matrix)):

        print(f"\n==============================")
        print(f"FOLD {fold+1}")
        print(f"==============================")

        # track best per fold
        fold_best_pol_f1 = -1
        fold_best_epoch = -1
        fold_best_found_macro = 0


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

            # save per-epoch data to logs
            epoch_logs.append({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "found_macro": found_macro,
                "found_micro": found_micro,
                "pol_macro_mean": mean_pol_macro,
                "pol_micro_mean": mean_pol_micro,
                "epoch_time": epoch_time,
                **{f"pol_macro_{f}": s for f, s in zip(FOUNDATIONS, pol_macro_set)},
                **{f"pol_micro_{f}": s for f, s in zip(FOUNDATIONS, pol_micro_set)},
            })

            # minimize printing
            print(f"\nEpoch {epoch+1}")
            print("="*50)
            print(f"{'Metric':<30} {'Score':>10}")
            print("-"*50)

            print(f"{'Train Loss':<30} {train_loss:>10.4f}")
            print(f"{'Foundation Macro F1':<30} {found_macro:>10.4f}")
            print(f"{'Polarity Macro F1 (avg)':<30} {mean_pol_macro:>10.4f}")
            print(f"{'Epoch Time (s)':<30} {epoch_time:>10.2f}")

            # update fold best (always per fold)
            if mean_pol_macro > fold_best_pol_f1:
                fold_best_pol_f1 = mean_pol_macro
                fold_best_found_macro = found_macro
                fold_best_epoch = epoch + 1
            
            # update global best
            if mean_pol_macro > best_pol_f1:
                best_pol_f1 = mean_pol_macro
                best_model_state = model.state_dict()
                best_fold = fold + 1

        fold_logs.append({
            "fold": fold + 1,
            "best_epoch": fold_best_epoch,
            "best_found_macro": fold_best_found_macro,
            "best_pol_macro": fold_best_pol_f1
        })

    # convert logs to df
    epoch_df = pd.DataFrame(epoch_logs)
    fold_df = pd.DataFrame(fold_logs)

    foundation_scores = fold_df["best_found_macro"]
    polarity_scores = fold_df["best_pol_macro"]

    print(f"\n==============================")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"==============================")

    print("Foundation Macro F1 per fold: ", foundation_scores)
    print("Mean Foundation Macro F1: ", np.mean(foundation_scores))

    print("Mean Polarity Macro F1 per fold: ", polarity_scores)
    print("Mean Polarity Macro F1 overall: ", np.mean(polarity_scores))
    print("\n")

    

    os.makedirs("results", exist_ok=True)
    epoch_df.to_csv("results/epoch_logs.csv", index=False)
    fold_df.to_csv("results/fold_summary.csv", index=False)

    summary_df = pd.DataFrame({
        "metric": ["foundation_macro", "polarity_macro"],
        "mean": [
            fold_df["best_found_macro"].mean(),
            fold_df["best_pol_macro"].mean()
        ],
        "std": [
            fold_df["best_found_macro"].std(),
            fold_df["best_pol_macro"].std()
        ]
    })

    summary_df.to_csv("results/summary.csv", index=False)

    # MODEL AND TOKENIZER SAVING
    os.makedirs(args.tokenizer_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    tokenizer.save_pretrained(args.tokenizer_dir)

    torch.save({
        "model_state_dict": best_model_state,
        "best_fold": best_fold,
        "best_pol_f1": best_pol_f1,
        "lambda_weight": args.lambda_weight,
        "max_len": args.max_len,
        "foundations": FOUNDATIONS
    }, os.path.join(args.model_dir, f"h_model_best.pt"))
    print(f"\nBest model saved from fold {best_fold} with Polarity Macro F1={best_pol_f1:.4f}.")


if __name__ == "__main__":
    main()