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
        default="data/lexicons/eMFD_wordlist.csv",
        help="Path to eMFD csv"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/lexicons",
        help="Output folder for combined lexicon CSV"
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

    return foundation_map.get(label.lower(), label.lower())

# =========================
# LOAD FILES
# =========================
def load_mfd(path):
    df = pd.read_csv(path)
    df["word"] = df["word"].str.lower()
    df["category"] = df["category"].apply(normalize_foundation)
    df["source"] = "mfd"

    return df

def load_mfd2(path):
    label_dict = {}
    rows = []
    
    # to delimit the section of the file, separated by "%"
    section = 0

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line ==  "%":
                section += 1
                continue

            if section == 1:
                parts = line.split()
                if len(parts) < 2:
                    continue

                cat_id = parts[0]
                label = parts[1]

                foundation, polarity = label.split(".")

                foundation = normalize_foundation(foundation)

                label_dict[cat_id] = foundation, polarity

            elif section == 2:
                parts = line.rsplit(maxsplit=1)

                if len(parts) < 2:
                    continue

                word = parts[0]
                cat_ids = parts[1].split()

                for cid in cat_ids:
                    foundation, polarity = label_dict[cid]
                    rows.append((word.lower(), foundation, polarity, "mfd2"))

    return pd.DataFrame(rows, columns=["word", "category", "sentiment", "source"])

def load_emfd(path, threshold):
    emfd = pd.read_csv(path)
    emfd_fixed = []

    for _, row in emfd.iterrows():
        for f in ["care", "fairness", "loyalty", "authority", "sanctity"]:
            found_col = f"{f}_p"
            pol_col = f"{f}_sent"
            if row[found_col] >= threshold:
                normal_label = normalize_foundation(f)
                
                if row[pol_col] >= 0:
                    sentiment_label = "virtue"
                else:
                    sentiment_label = "vice"

                emfd_fixed.append((row["word"].lower(), normal_label, sentiment_label, "emfd"))
                

    return pd.DataFrame(emfd_fixed, columns=["word", "category", "sentiment", "source"])

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

    print("Loading MFD...")
    mfd = load_mfd(args.mfd)

    print("Loading MFD 2.0...")
    mfd2 = load_mfd2(args.mfd2)

    print("Loading eMFD...")
    threshold = float(args.threshold)
    emfd = load_emfd(args.emfd, threshold)

    combined_lexicon = pd.concat([mfd, mfd2, emfd], axis=0, ignore_index=True)
    combined_lexicon.drop_duplicates(inplace=True)

    combined_lexicon.to_csv(os.path.join(args.output, "combined_lexicon.csv"), index=False)

    print("Saved combined lexicons.")

if __name__ == "__main__":
    main()