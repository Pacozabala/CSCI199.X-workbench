#!/usr/bin/env python3

"""
BoW-based Virtue/Vice Polarity Assignment for MFRC

Default input directory: data/

Example usage:

python bow_polarity_assignment.py \
    --output_path outputs/MFRC_polarities.csv \
    --use_lemma \
    --use_frequency \
    --tie_break drop \
    --drop_zero_signal
"""

import os
import argparse
import pandas as pd

DEFAULT_DATA_DIR = "data"
DEFAULT_MFD = os.path.join(DEFAULT_DATA_DIR, "lexicons/MFD_original.csv")
DEFAULT_MFRC = os.path.join(DEFAULT_DATA_DIR, "final_mfrc_data.csv")


# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign virtue/vice polarity labels to MFRC using MFD (BoW baseline)"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/MFRC_polarities.csv",
        help="Path to save output CSV"
    )

    parser.add_argument(
        "--use_nltk",
        action="store_true",
        help="Use NLTK TweetTokenizer before scoring"
    )

    parser.add_argument(
        "--use_lemma",
        action="store_true",
        help="Use spaCy lemmatization"
    )

    parser.add_argument(
        "--use_frequency",
        action="store_true",
        help="Use frequency counting instead of set overlap"
    )

    parser.add_argument(
        "--tie_break",
        type=str,
        default="virtue",
        choices=["virtue", "vice", "neutral", "drop"],
        help="How to handle alpha == 0 (default: virtue)"
    )

    parser.add_argument(
        "--drop_zero_signal",
        action="store_true",
        help="Drop rows where no moral words were found"
    )

    return parser.parse_args()


# -----------------------------
# Normalize Foundations
# -----------------------------

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


# -----------------------------
# Main
# -----------------------------

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
    FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]

    def get_word_set(category, sentiment):
        return set(
            MFD[
                (MFD["category"] == category) &
                (MFD["sentiment"] == sentiment)
            ]["word"].str.lower()
        )

    lexicon = {
        f: {
            "virtue": get_word_set(f, "virtue"),
            "vice": get_word_set(f, "vice")
        } for f in FOUNDATIONS
    }

    # Optional NLP tools
    if args.use_nltk:
        print("Using NLTK TweetTokenizer...")
        import nltk
        from nltk.tokenize import TweetTokenizer

        tokenizer = TweetTokenizer(
            preserve_case=False,
            reduce_len=True,
            strip_handles=False
        )

        # MFRC["text"] = MFRC["text"].apply(
        #     lambda x: " ".join(tokenizer.tokenize(str(x)))
        # )
    
    if args.use_lemma:
        import spacy
        nlp = spacy.load("en_core_web_sm")


    # Polarity Assignment Function
    def assign_polarity_label(text, annotation):

        foundations = annotation.lower().split(",")
        text = str(text)

        if args.use_nltk:
            tokens = tokenizer.tokenize(text)
        elif args.use_lemma:
            doc = nlp(text)
            tokens = [token.lemma_.lower() for token in doc]
        else:
            tokens = text.lower().split()

        results = []

        for foundation in foundations:
            foundation = foundation.strip()

            if foundation not in FOUNDATIONS:
                continue

            virtue_set = lexicon[foundation]["virtue"]
            vice_set = lexicon[foundation]["vice"]

            if args.use_frequency:
                freq_virtue = sum(t in virtue_set for t in tokens)
                freq_vice = sum(t in vice_set for t in tokens)
            else:
                token_set = set(tokens)
                freq_virtue = len(token_set & virtue_set)
                freq_vice = len(token_set & vice_set)

            alpha = freq_virtue - freq_vice

            # handle zero signal
            if freq_virtue == 0 and freq_vice == 0:
                if args.drop_zero_signal:
                    continue
            
            # tie-breaking
            if alpha > 0:
                label = "virtue"
            elif alpha < 0:
                label = "vice"
            else:
                if args.tie_break == "virtue":
                    label = "virtue"
                if args.tie_break == "vice":
                    label = "vice"
                elif args.tie_break == "neutral":
                    label = "neutral"
                elif args.tie_break == "drop":
                    continue

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