#!/usr/bin/env python3

"""
BoW-based Virtue/Vice Polarity Assignment for MFRC
Using Moral Foundations Dictionary (MFD)

Command line usage:

python bow_polarity_assignment.py \
    --mfd_path data/MFD_original.csv \
    --mfrc_path data/final_mfrc_data.csv \
    --output_path data/MFRC_polarities.csv \
    --use_nltk
"""

import os
import argparse
import pandas as pd


# =============================
# Argument Parser
# =============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign virtue/vice polarity labels to MFRC using MFD (BoW baseline)"
    )

    parser.add_argument(
        "--mfd_path",
        type=str,
        required=True,
        help="Path to MFD CSV file"
    )

    parser.add_argument(
        "--mfrc_path",
        type=str,
        required=True,
        help="Path to MFRC CSV file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output CSV"
    )

    parser.add_argument(
        "--use_nltk",
        action="store_true",
        help="Use NLTK TweetTokenizer before scoring"
    )

    return parser.parse_args()


# =============================
# Normalize Foundations
# =============================

def normalize_foundations(annotation):
    foundations = annotation.split(',')

    foundations = [
        f.replace('Equality', 'Fairness')
         .replace('Proportionality', 'Fairness')
         .replace('Loyalty', 'Ingroup')
         .replace('Care', 'Harm')
        for f in foundations
    ]

    return ','.join(set(foundations))


# =============================
# Main
# =============================

def main():

    args = parse_args()

    print("Loading MFD...")
    MFD = pd.read_csv(args.mfd_path)
    print("Loading MFRC...")
    MFRC = pd.read_csv(args.mfrc_path)

    # normalize foundations
    MFRC["annotation"] = MFRC["annotation"].apply(normalize_foundations)

    # remove Non-Moral
    MFRC = MFRC[MFRC["annotation"] != "Non-Moral"]

    # remove Not Confident
    MFRC = MFRC[MFRC["confidence"] != "Not Confident"]

    # build lexicon sets
    FOUNDATIONS = set(["authority", "fairness", "harm", "ingroup", "purity"])

    def get_word_set(category, sentiment):
        return set(
            MFD[
                (MFD["category"] == category) &
                (MFD["sentiment"] == sentiment)
            ]["word"].str.lower()
        )

    authority_virtue = get_word_set("authority", "virtue")
    authority_vice = get_word_set("authority", "vice")

    fairness_virtue = get_word_set("fairness", "virtue")
    fairness_vice = get_word_set("fairness", "vice")

    harm_virtue = get_word_set("harm", "virtue")
    harm_vice = get_word_set("harm", "vice")

    ingroup_virtue = get_word_set("ingroup", "virtue")
    ingroup_vice = get_word_set("ingroup", "vice")

    purity_virtue = get_word_set("purity", "virtue")
    purity_vice = get_word_set("purity", "vice")

    # Optional NLTK tokenization
    if args.use_nltk:
        print("Using NLTK TweetTokenizer...")
        import nltk
        from nltk.tokenize import TweetTokenizer

        tokenizer = TweetTokenizer(
            preserve_case=False,
            reduce_len=True,
            strip_handles=False
        )

        MFRC["text"] = MFRC["text"].apply(
            lambda x: " ".join(tokenizer.tokenize(str(x)))
        )

    # Polarity Assignment Function
    def assign_polarity_label(text, annotation):

        foundations = annotation.lower().split(",")
        words = set(str(text).lower().split())

        results = []

        for foundation in foundations:
            foundation = foundation.strip()

            if foundation not in FOUNDATIONS:
                continue

            if foundation == "authority":
                virtue_set = authority_virtue
                vice_set = authority_vice

            elif foundation == "fairness":
                virtue_set = fairness_virtue
                vice_set = fairness_vice

            elif foundation == "harm":
                virtue_set = harm_virtue
                vice_set = harm_vice

            elif foundation == "ingroup":
                virtue_set = ingroup_virtue
                vice_set = ingroup_vice

            elif foundation == "purity":
                virtue_set = purity_virtue
                vice_set = purity_vice

            freq_virtue = len(words & virtue_set)
            freq_vice = len(words & vice_set)

            alpha = freq_virtue - freq_vice

            if alpha >= 0:
                label = "virtue"
            else:
                label = "vice"

            results.append(f"{foundation}.{label}")

        return results

    # -----------------------------
    # Score Dataset
    # -----------------------------
    print("Assigning polarity labels...")

    MFRC["polarity"] = MFRC.apply(
        lambda row: ",".join(
            assign_polarity_label(row["text"], row["annotation"])
        ),
        axis=1
    )

    # -----------------------------
    # Save Output
    # -----------------------------
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    MFRC.to_csv(args.output_path, index=False)

    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()