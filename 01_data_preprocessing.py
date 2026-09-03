"""
01_data_preprocessing.py
-------------------------
Loads ODIR-5K's full_df.csv / data.xlsx, builds:
  1. Multi-label disease targets (N, D, G, C, A, H, M) — excluding 'Other' (O) images
  2. DR severity labels extracted from free-text diagnostic keywords
  3. A patient-wise train/val/test split (avoids leaking the same patient's
     left/right eye across splits)

Output: data/train.csv, data/val.csv, data/test.csv
Each row = one eye image, with columns:
  filename, N,D,G,C,A,H,M (0/1 each), dr_severity (0-4 or -1 if not DR/unknown)
"""

import argparse
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

DISEASE_COLS = ["N", "D", "G", "C", "A", "H", "M"]

# Ordinal DR severity vocabulary, most severe phrase checked first
DR_SEVERITY_PATTERNS = [
    (4, r"proliferative retinopathy"),
    (3, r"severe non ?proliferative retinopathy|severe npdr"),
    (2, r"moderate non ?proliferative retinopathy|moderate npdr"),
    (1, r"mild non ?proliferative retinopathy|mild npdr"),
]


def extract_dr_severity(keyword_text: str) -> int:
    """Rule-based severity extraction from ODIR diagnostic-keyword text.
    Returns 0-4 for DR-related phrases found, or -1 if no DR severity
    phrase is present (i.e. not applicable / not stated)."""
    if not isinstance(keyword_text, str):
        return -1
    text = keyword_text.lower()
    for severity, pattern in DR_SEVERITY_PATTERNS:
        if re.search(pattern, text):
            return severity
    if "diabetic retinopathy" in text:
        return 0  # DR mentioned but no explicit grade -> treat as mild/unspecified
    return -1


def build_long_format(df: pd.DataFrame, exclude_other: bool = True) -> pd.DataFrame:
    """ODIR dataset has one row per PATIENT with separate left/right
    columns. We convert to one row per EYE IMAGE since that's our model's
    input unit, while keeping a patient_id column for grouped splitting.
    
    If exclude_other=True, records where the 'Other' (O) disease column is 1
    are excluded."""
    rows = []
    excluded_other_count = 0

    for _, r in df.iterrows():
        # Check if the patient/record has the 'Other' (O) label
        is_other = False
        if "O" in r and not pd.isna(r["O"]):
            try:
                is_other = (int(r["O"]) == 1)
            except (ValueError, TypeError):
                is_other = False

        if exclude_other and is_other:
            excluded_other_count += 1
            continue

        for side in ["Left", "Right"]:
            fname = r.get(f"{side}-Fundus")
            keywords = r.get(f"{side}-Diagnostic Keywords", "")
            if not isinstance(fname, str) or fname.strip() == "":
                continue
            row = {
                "patient_id": r["ID"],
                "filename": fname,
                "age": r.get("Age", None),
                "sex": r.get("Sex", None),
                "keywords": keywords,
            }
            for d in DISEASE_COLS:
                row[d] = int(r[d]) if d in r and not pd.isna(r[d]) else 0
            row["dr_severity"] = extract_dr_severity(keywords) if row["D"] == 1 else -1
            rows.append(row)

    if exclude_other:
        print(f"Excluded {excluded_other_count} patient records with disease 'Other' (O=1).")

    return pd.DataFrame(rows)


def patient_wise_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42):
    patient_ids = df["patient_id"].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=seed)
    train_ids, val_ids = train_test_split(
        train_ids, test_size=val_size / (1 - test_size), random_state=seed
    )
    train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True, help="path to ODIR full_df.csv or data.xlsx")
    ap.add_argument("--img_dir", required=True, help="folder containing the fundus images")
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--include_other", action="store_true", help="Set to include 'Other' disease images")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.csv_path.endswith(".xlsx") or args.csv_path.endswith(".xls"):
        df = pd.read_excel(args.csv_path)
    else:
        df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} patient rows")

    exclude_other = not args.include_other
    long_df = build_long_format(df, exclude_other=exclude_other)
    print(f"Expanded to {len(long_df)} eye-image rows (7 disease classes: {DISEASE_COLS})")

    # Drop rows whose image file doesn't actually exist on disk
    long_df["exists"] = long_df["filename"].apply(
        lambda f: os.path.isfile(os.path.join(args.img_dir, f))
    )
    missing = (~long_df["exists"]).sum()
    if missing:
        print(f"WARNING: {missing} listed images not found in {args.img_dir}, dropping them")
    long_df = long_df[long_df["exists"]].drop(columns=["exists"])

    print("\nClass distribution (positive counts):")
    print(long_df[DISEASE_COLS].sum())

    print("\nDR severity distribution (only DR-positive eyes):")
    print(long_df.loc[long_df["D"] == 1, "dr_severity"].value_counts().sort_index())

    train_df, val_df, test_df = patient_wise_split(long_df)
    print(f"\nSplit sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    print(f"\nSaved train.csv / val.csv / test.csv to {args.out_dir}")


if __name__ == "__main__":
    main()