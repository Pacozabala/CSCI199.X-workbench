'''
Sample usage:

python scripts/prep_multi_data.py
'''

import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]

# Args Parser
'''
Argument Parser looks for 4 arguments
- input: path to input file
- output: path where output CSV is created
- seed: randomness seed for dataset splitting
- test-size: ratio for training and test sets
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prep hierarchical dataset (5 foundation + 5 polarity heads)."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/MFRC_polarity.csv",
        help="Path to input CSV file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/hierarchical_dataset",
        help="Output directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    return parser.parse_args()

# Main Method
'''
Main method:
- reads input
- builds columns for each classifier head
- saves output df to CSV
'''
def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df[["text", "polarity"]].copy()
    df = df.dropna(subset=["polarity"])

    # 1. Split multi-label rows
    df["polarity"] = df["polarity"].str.split(",")
    df = df.explode("polarity")
    df["polarity"] = df["polarity"].str.strip()

    # 2. Parse foundation + pole
    df[["foundation", "pole"]] = df["polarity"].str.split(".", expand=True)

    # 3. Aggregate per text
    texts = df["text"].unique()
    multi_df = pd.DataFrame({"text": texts})

    for f in FOUNDATIONS:
        # foundation presence
        f_texts = df[df["foundation"] == f]["text"].unique()
        multi_df[f] = multi_df["text"].isin(f_texts).astype(int)

        # polarity (default neutral = 2)
        multi_df[f"{f}_pol"] = 2

        # assign virtue (0)
        virtue_texts = df[
            (df["foundation"] == f) & (df["pole"] == "virtue")
        ]["text"].unique()
        multi_df.loc[multi_df["text"].isin(virtue_texts), f"{f}_pol"] = 0

        # assign vice (1)
        vice_texts = df[
            (df["foundation"] == f) & (df["pole"] == "vice")
        ]["text"].unique()
        multi_df.loc[multi_df["text"].isin(vice_texts), f"{f}_pol"] = 1

    # 4. Train/Val/Test Split
    train_df, temp_df = train_test_split(
        multi_df,
        test_size=args.test_size,
        random_state=args.seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=args.seed,
    )

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # --------------------------
    # 5. Save unified splits
    # --------------------------
    train_df.to_csv(os.path.join(args.output, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output, "test.csv"), index=False)

    print("Saved hierarchical dataset.")
    print("Columns:")
    print(train_df.columns.tolist())


if __name__ == "__main__":
    main()