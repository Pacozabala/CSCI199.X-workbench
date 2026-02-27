import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch.optim import AdamW
from sklearn.metrics import f1_score

# =========================
# ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        default="data/hierarchical_dataset",
                        help="Base dataset directory")

    parser.add_argument("--output_dir", type=str,
                        default="outputs",
                        help="Where to save models")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)

    return parser.parse_args()


# =========================
# DATASET
# =========================
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.foundation_cols = ["authority","fairness","harm","ingroup","purity"]
        self.polarity_cols = [f"{f}_pol" for f in self.foundation_cols]
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

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

        foundation_labels = torch.tensor(
            self.df.iloc[idx][self.foundation_cols].values,
            dtype=torch.float
        )

        polarity_labels = torch.tensor(
            self.df.iloc[idx][self.polarity_cols].values,
            dtype=torch.long
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "foundation_labels": foundation_labels,
            "polarity_labels": polarity_labels
        }

# =========================
# MODEL
# =========================
class HierarchicalRoBERTa(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super().__init__()

        self.lambda_weight = lambda_weight
        self.encoder = RobertaModel.from_pretrained("roberta-base")

        hidden = self.encoder.config.hidden_size

        # 5 foundation binary heads
        self.foundation_heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(5)]
        )

        # 5 polarity ternary heads
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

        pooled = outputs.last_hidden_state[:, 0]  # CLS token

        foundation_logits = torch.cat(
            [head(pooled) for head in self.foundation_heads],
            dim=1
        )

        polarity_logits = torch.stack(
            [head(pooled) for head in self.polarity_heads],
            dim=1
        )  # [batch, 5, 3]

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

            loss = foundation_loss + self.lambda_weight * polarity_loss

        return {
            "loss": loss,
            "foundation_logits": foundation_logits,
            "polarity_logits": polarity_logits
        }

# =========================
# TRAIN / EVAL
# =========================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        foundation_labels = batch["foundation_labels"].to(device)
        polarity_labels = batch["polarity_labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            foundation_labels=foundation_labels,
            polarity_labels=polarity_labels
        )

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    true = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            preds = (torch.sigmoid(logits) > 0.5)
            true.extend(batch["labels"].numpy())

    return f1_score(true, preds, average="macro")


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_folder = "multi_model"
    data_path = os.path.join(args.data_dir, label_folder)

    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = TextDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        args.max_len
    )

    val_dataset = TextDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
        args.max_len
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size)

    model = HierarchicalRoBERTa(lambda_weight=1.0)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(f"Training {label_folder}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_f1 = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Macro F1: {val_f1:.4f}")

    # Save model
    save_path = os.path.join(args.output_dir, label_folder)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()