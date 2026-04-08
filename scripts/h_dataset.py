import torch
from torch.utils.data import Dataset

FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]

# =========================
# DATASET
# =========================
class HierarchicalDataset(Dataset):
    def __init__(self, df, encodings, indices):
        self.indices = indices
        self.encodings = encodings

        self.foundation_labels = torch.tensor(
            df[FOUNDATIONS].astype(float).values,
            dtype=torch.float
        )

        self.polarity_labels = torch.tensor(
            df[[f"{f}_pol" for f in FOUNDATIONS]].astype(int).values,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        return {
            "input_ids": self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
            "foundation_labels": self.foundation_labels[i],
            "polarity_labels": self.polarity_labels[i]
        }