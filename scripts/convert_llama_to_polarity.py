"""Convert LLaMA MFRC outputs to `text` + comma-separated `polarity` labels.

Sample usage:

python scripts/convert_llama_to_polarity.py \
  --input_path data/llama_zeroshot_data.csv \
  --output_path data/MFRC_multi_llama_zeroshot.csv

python scripts/convert_llama_to_polarity.py \
  --input_path data/llama_fewshot_data.csv \
  --output_path data/MFRC_multi_llama_fewshot.csv \
  --drop_not_confident
"""

import argparse

LLAMA_TO_POLARITY = {
    "care": "harm.virtue",
    "harm": "harm.vice",
    "fairness": "fairness.virtue",
    "cheating": "fairness.vice",
    "loyalty": "ingroup.virtue",
    "betrayal": "ingroup.vice",
    "authority": "authority.virtue",
    "subversion": "authority.vice",
    "purity": "purity.virtue",
    "degradation": "purity.vice",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert llama_{zeroshot,fewshot}_data.csv files into the"
            " polarity format consumed by scripts/prep_multi_data.py"
        )
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to llama_zeroshot_data.csv or llama_fewshot_data.csv",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to output CSV (e.g., data/MFRC_multi_llama_zeroshot.csv)",
    )
    parser.add_argument(
        "--drop_not_confident",
        action="store_true",
        help="Drop rows where confidence equals 'Not Confident'",
    )
    return parser.parse_args()


def _active_to_polarity(row):
    labels = [
        mapped_label
        for llama_col, mapped_label in LLAMA_TO_POLARITY.items()
        if int(row[llama_col]) == 1
    ]
    return ",".join(labels)


def main():
    args = parse_args()
    # local import so script can still be syntax-checked in environments
    # where optional dependencies are not installed.
    import pandas as pd

    df = pd.read_csv(args.input_path)

    required_columns = {"text", "confidence", *LLAMA_TO_POLARITY.keys()}
    missing = required_columns - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in input file: {missing_cols}")

    if args.drop_not_confident:
        df = df[df["confidence"].str.lower() != "not confident"]

    out_df = df.copy()
    out_df["polarity"] = out_df.apply(_active_to_polarity, axis=1)
    out_df = out_df[out_df["polarity"] != ""].copy()

    # Keep at least text + polarity (plus extras for traceability).
    output_columns = ["text", "polarity", "confidence"]
    existing_output_columns = [c for c in output_columns if c in out_df.columns]
    out_df = out_df[existing_output_columns]

    out_df.to_csv(args.output_path, index=False)

    print(f"Saved: {args.output_path}")
    print("Rows:", len(out_df))


if __name__ == "__main__":
    main()
