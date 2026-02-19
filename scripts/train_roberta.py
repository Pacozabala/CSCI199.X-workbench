import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 2

train_df = pd.read_csv("data/binary_datasets/authority_vice/train.csv")
val_df = pd.read_csv("data/binary_datasets/authority_vice/val.csv")
test = pd.read_csv("data/binary_datasets/authority_vice/test.csv")

class TextDataset(Dataset):
    def __init__(self,texts,labels,tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation = True,
            padding = "max_length",
            max_length = MAX_LEN,
            return_tensors = "pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

train_dataset = TextDataset(
    train_df["text"].tolist(),
    train_df["label"].tolist(),
    tokenizer
)

val_dataset = TextDataset(
    val_df["text"].tolist(),
    val_df["label"].tolist(),
    tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds = []
    true = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            )

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            preds.extend(predictions)
            true.extend(batch["labels"].numpy())
    
    return f1_score(true, preds)


for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader)
    val_f1 = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val F1: {val_f1:.4f}")