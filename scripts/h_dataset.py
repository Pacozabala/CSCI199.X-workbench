import torch
from torch.utils.data import Dataset

FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]

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