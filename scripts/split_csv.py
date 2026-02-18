import pandas as pd
import os

# ==========================
# CONFIG
# ==========================
INPUT_CSV = "data/MFRC_polarities.csv"
OUTPUT_DIR = "../data/lexicons/binary_datasets"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# 1. Load CSV, get columns
# ==========================
df = pd.read_csv(INPUT_CSV)

df = df[["text", "polarity"]].copy()
df = df.dropna(subset=["polarity"])

# ==========================
# 2. Split multi-label rows
# ==========================
df["polarity"] = df["polarity"].str.split(",")
df = df.explode("polarity")

df["polarity"] = df["polarity"].str.strip()

df = df.drop_duplicates()

# ==========================
# 3. Get unique labels
# ==========================
all_labels = sorted(df["polarity"].unique())
print(f"Found {len(all_labels)} unique labels:")
print(all_labels)

# ==========================
# 4. Build binary datasets
# ==========================
all_texts = df["text"].unique()

for label in all_labels:
    print(f"Processing label:{label}")

    # texts that contain the label
    positive_texts = df[df["polarity"] == label]["text"].unique()

    # build binary dataset
    binary_df = pd.DataFrame({
        "text": all_texts
    })

    binary_df["label"] = binary_df["text"].isin(positive_texts).astype(int)

    # remove duplicate binary rows
    binary_df = binary_df.drop_duplicates()

    # save file
    safe_label = label.replace(".","_")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_label}.csv")
    binary_df.to_csv(output_path, index=False)
    print(f"Finished dataset for foundation; {safe_label}")

print("Finished.")