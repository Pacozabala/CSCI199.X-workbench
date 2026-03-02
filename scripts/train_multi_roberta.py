import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from sklearn.metrics import f1_score

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
# DATASET
# =========================
class HierarchicalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.foundation_labels = torch.tensor(
            df[FOUNDATIONS].astype(float).values,
            dtype=torch.float
        )

        self.polarity_labels = torch.tensor(
            df[[f"{f}_pol" for f in FOUNDATIONS]].astype(int).values,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "foundation_labels": self.foundation_labels[idx],
            "polarity_labels": self.polarity_labels[idx]
        }
    
# =========================
# MODEL
# =========================
'''
Defines the Hierarchical Model, which inherits nn.Module
- forward() function: used in training, returns loss and logits.
''' 
class HierarchicalRoBERTa(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super().__init__()

        self.lambda_weight = lambda_weight
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        hidden = self.encoder.config.hidden_size

        self.foundation_heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(5)]
        )

        self.polarity_heads = nn.ModuleList(
            [nn.Linear(hidden, 3) for _ in range(5)]
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input_ids, attention_mask,
                foundation_labels=None,
                polarity_labels=None):
        
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # get the 1st (CLS) token
        pooled = outputs.last_hidden_state[:,0]

        # flatten all found. logits into one array
        foundation_logits = torch.cat(
            [head(pooled) for head in self.foundation_heads],
            dim=1
        )


        polarity_logits = torch.stack(
            [head(pooled) for head in self.polarity_heads],
            dim=1
        ) # constructs the 3D array with size: [batch, 5, 3] -> batch entries, 5 triples per entry

        loss = None

        if foundation_labels is not None:

            foundation_loss = self.bce(
                foundation_logits,
                foundation_labels
            )

            polarity_loss = 0
            for f in range(5):
                mask = foundation_labels[:, f]
                ce_loss = self.ce(
                    polarity_logits[:, f, :],
                    polarity_labels[:, f]
                )
                
                masked_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)
                polarity_loss += masked_loss
            
            # masked overall loss
            loss = foundation_loss + (self.lambda_weight* polarity_loss)

        return loss, foundation_logits, polarity_logits
    
# =========================
# TRAINING FUNCTION
# =========================
'''
Trains the model for 1 epoch.
'''
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        foundation_labels = batch["foundation_labels"].to(device)
        polarity_labels = batch["polarity_labels"].to(device)

        # gets rid of gradient from previous pass
        optimizer.zero_grad()

        # this calls forward()
        loss, _, _ = model(
            input_ids,
            attention_mask,
            foundation_labels,
            polarity_labels
        )

        # backpropagate the loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# EVALUATION
# =========================
'''
Evaluates the model.
Returns foundation f1, polarity f1 per foundation, and mean polarity f1
'''
def evaluate(model, loader, device):
    model.eval()

    all_found_preds = []
    all_found_true = []

    all_pol_preds = [[] for _ in range(5)]
    all_pol_true = [[] for _ in range(5)]

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            foundation_labels = batch["foundation_labels"].to(device)
            polarity_labels = batch["polarity_labels"].to(device)

            _, foundation_logits, polarity_logits = model(
                input_ids, attention_mask
            )

            # foundation predictions
            found_preds = (torch.sigmoid(foundation_logits) > 0.5)

            all_found_preds.append(found_preds.cpu())
            all_found_true.append(foundation_labels.cpu())

            # polarity predictions (masked)
            pol_preds = torch.argmax(polarity_logits, dim=2)

            for f in range(5):
                mask_f = found_preds[:, f] == 1 # predicted foundations mask

                if mask_f.sum() > 0:
                    preds_f = pol_preds[mask_f, f]
                    true_f = polarity_labels[mask_f, f]

                    all_pol_preds[f].append(preds_f.cpu())
                    all_pol_true[f].append(true_f.cpu())
        
    foundation_f1 = f1_score(
        torch.cat(all_found_true).numpy().flatten(),
        torch.cat(all_found_preds).numpy().flatten(),
        average="macro"
    )

    polarity_f1_scores = []
    for f in range(5):
        if len(all_pol_true[f]) > 0:
            y_true = torch.cat(all_pol_true[f]).numpy()
            y_pred = torch.cat(all_pol_preds[f]).numpy()

            f1 = f1_score(y_true, y_pred, average="macro")
            polarity_f1_scores.append(f1)
        else:
            polarity_f1_scores.append(0.0)
    
    mean_polarity_f1 = sum(polarity_f1_scores) / 5

    return foundation_f1, polarity_f1_scores, mean_polarity_f1

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
        train_loss = train_epoch(model, train_loader, optimizer, device)
        foundation_f1, polarity_f1_set, mean_polarity_f1 = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Foundation Macro F1: {foundation_f1:.4f}")
        for f in range(5):
            print(f"{FOUNDATIONS[f]} Polarity Macro F1 (masked): {polarity_f1_set[f]:.4f}")
        print(f"Mean Polarity Macro F1 (masked): {mean_polarity_f1:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "hierarchical_model.pt"))

    print("\nModel saved.")


if __name__ == "__main__":
    main()