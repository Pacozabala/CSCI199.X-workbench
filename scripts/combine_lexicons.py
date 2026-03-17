import csv
import os
import argparse
import pandas as pd

# =========================
# ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine lexicons"
    )

    parser.add_argument(
        "--mfd",
        type=str,
        default="data/lexicons/MFD_original.csv",
        help="Path to MFD csv"
    )

    parser.add_argument(
        "--mfd2",
        type=str,
        default="data/lexicons/mfd2.0.dic",
        help="Path to MFD 2.0 dic"
    )

    parser.add_argument(
        "--emfd",
        type=str,
        default="data/lexicons/eMFD_worldist.csv",
        help="Path to eMFD csv"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/lexicons/combined_lexcion.csv",
        help="Output path for combined lexicon CSV"
    )

    parser.add_argument(
        "--threshold",
        type=str,
        default="0.1",
        help="Threshold value for probability labels"
    )

    return parser.parse_args()

# =========================
# FOUNDATION NORMALIZATION AND MAPPING
# =========================
FOUNDATIONS = ["authority", "fairness", "harm", "ingroup", "purity"]

def normalize_foundation(label):
    foundation_map = {
        "care": "harm",
        "harm": "harm",
        "fairness": "fairness",
        "cheating": "fairness",
        "loyalty": "ingroup",
        "betrayal": "ingroup",
        "ingroup": "ingroup",
        "authority": "authority",
        "subversion": "authority",
        "sanctity": "purity",
        "degradation": "purity",
        "purity": "purity"
    }

    return foundation_map[label]

# =========================
# LOAD FILES
# =========================
def load_mfd(path):
    df = pd.read_csv(path)
    df["category"] = df["category"].apply(normalize_foundation, axis=1)
    df["source"] = "mfd"

    return df

def load_mfd2(path):
    label_dict = {}
    rows = [("word", "category", "sentiment", "source")]
    mode = "categories"

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            if line ==  "%":
                if mode == "categories":
                    mode = "words"
                    continue

            if mode == "categories":
                parts = line.split()

                cat_id = parts[0]
                label = parts[1]

                foundation, polarity = label.split(".")

                foundation = normalize_foundation(foundation)

                label_dict[cat_id] = foundation, polarity

            if mode == "words":
                parts = line.split()

                word = parts[0]
                cat_ids = parts[1:]

                for id in cat_ids:
                    foundation, polarity = label_dict[id]
                    rows.append((word, foundation, polarity, "mfd2"))

def load_emfd(path, threshold):
    emfd = pd.read_csv(path)
    emfd_fixed = [("word", "category", "sentiment", "source")]

    for _, row in emfd.iterrows():
        for f in ["care", "fairness", "loyalty", "authority", "sanctity"]:
            if row[f"{f}_p"] >= threshold:
                normal_label = normalize_foundation(f)
                
                if row[f"{f}_sent"] >= 0:
                    sentiment_label = "virtue"
                else:
                    sentiment_label = "vice"

                emfd_fixed.append((row["word"].lower(), normal_label, sentiment_label, "emfd"))
                

    return pd.DataFrame(emfd_fixed)

# =========================
# MAIN METHOD
# =========================
def main():
    '''
    Main method:
    - reads the 3 lexicons
    - converts mfd2.0.dic to csv
    - concatenates them
    '''
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    mfd = pd.read_csv(os.path.join(args.input, "MFD_original.csv"))
    mfd_2 = pd.read_csv(os.path.join(args.input, "mfd2.0.csv"))
    emfd = pd.read_csv(os.path.join(args.input, "eMFD_wordlist.csv"))