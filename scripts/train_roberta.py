import os
import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score

# =========================
# ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--foundation", type=str, required=True,
                        help="Moral foundation (e.g., authority)")
    parser.add_argument("--pole", type=str, required=True,
                        help="Polarity (e.g., vice or virtue)")

    parser.add_argument("--data_dir", type=str,
                        default="data/binary_datasets",
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
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
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

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
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
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
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

            preds.extend(predictions)
            true.extend(batch["labels"].numpy())

    return f1_score(true, preds, average="macro")


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_folder = f"{args.foundation}_{args.pole}"
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

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

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