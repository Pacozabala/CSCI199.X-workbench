import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

# ==========================
# Argument Parser
# ==========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prep binary datasets for 10 independent RoBERTa classifiers."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../data/MFRC_polarities.csv",
        help="Path to input CSV file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/binary_datasets",
        help="Output directory for binary datasets",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion reserved for val+test split (default 0.2 → 80/10/10)",
    )

    return parser.parse_args()


# ==========================
# Main
# ==========================
def main():
    args = parse_args()

    INPUT_CSV = args.input
    OUTPUT_DIR = args.output
    RANDOM_STATE = args.seed
    TEST_SIZE = args.test_size

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Input file: {INPUT_CSV}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Seed: {RANDOM_STATE}")
    print(f"Test size (val+test): {TEST_SIZE}")

    # ==========================
    # 1. Load CSV
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
    # 3. Build multi-hot matrix
    # ==========================
    all_texts = df["text"].unique()
    all_labels = sorted(df["polarity"].unique())

    print(f"Found {len(all_labels)} labels:")
    print(all_labels)

    multi_df = pd.DataFrame({"text": all_texts})

    for label in all_labels:
        positive_texts = df[df["polarity"] == label]["text"].unique()
        multi_df[label] = multi_df["text"].isin(positive_texts).astype(int)

    # ==========================
    # 4. Train / Val / Test Split
    # ==========================
    train_df, temp_df = train_test_split(
        multi_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # ==========================
    # 5. Save per-label datasets
    # ==========================
    for label in all_labels:
        safe_label = label.replace(".", "_")
        label_dir = os.path.join(OUTPUT_DIR, safe_label)
        os.makedirs(label_dir, exist_ok=True)

        train_label = train_df[["text", label]].rename(columns={label: "label"})
        val_label = val_df[["text", label]].rename(columns={label: "label"})
        test_label = test_df[["text", label]].rename(columns={label: "label"})

        train_label.to_csv(os.path.join(label_dir, "train.csv"), index=False)
        val_label.to_csv(os.path.join(label_dir, "val.csv"), index=False)
        test_label.to_csv(os.path.join(label_dir, "test.csv"), index=False)

        print(f"Saved splits for: {safe_label}")

    print("Finished.")


if __name__ == "__main__":
    main()
