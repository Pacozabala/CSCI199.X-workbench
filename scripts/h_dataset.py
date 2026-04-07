import torch
from torch.utils.data import Dataset

FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]

# =========================
# DATASET
# =========================
class HierarchicalDataset(Dataset):
    def __init__(self, df, encodings, max_len):
        self.texts = df["text"].tolist()
        self.encodings = encodings
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

        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "foundation_labels": self.foundation_labels[idx],
            "polarity_labels": self.polarity_labels[idx]
        }