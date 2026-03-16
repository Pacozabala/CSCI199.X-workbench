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
        "authority": "authority",
        "subversion": "authority",
        "sanctity": "purity",
        "degradation": "purity"
    }

    return foundation_map[label]


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