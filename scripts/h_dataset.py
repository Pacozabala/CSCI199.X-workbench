import torch
from torch.utils.data import Dataset

FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]

# =========================
# DATASET
# =========================
class HierarchicalDataset(Dataset):
    def __init__(self, df, encodings, indices):
        self.texts = df["text"].tolist()
        self.encodings = encodings
        self.indices = indices

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
        real_idx = self.indices[idx]

        return {
            "input_ids": self.encodings["input_ids"][real_idx],
            "attention_mask": self.encodings["attention_mask"][real_idx],
            "foundation_labels": self.foundation_labels[real_idx],
            "polarity_labels": self.polarity_labels[real_idx]
        }