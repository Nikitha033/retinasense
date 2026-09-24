"""
01_data_preprocessing.py
-------------------------
Loads ODIR-5K dataset (data.xlsx / full_df.csv), performs:
  1. Eye-level multi-label disease parsing (D, G, C, A, H, M) based on
     eye-specific diagnostic keywords.
  2. Genuinely normal eye preservation (N=1, all 6 disease targets=0) as negative screening samples.
  3. Individual eye-level exclusion of ambiguous "Other" findings (when neither target disease nor normal).
  4. Correct rule-based DR severity extraction (0=Unspecified, 1=Mild, 2=Moderate, 3=Severe, 4=PDR, -1=Non-DR).
  5. Patient-wise stratified train/val/test splitting to strictly prevent patient eye leakage.

Output: data/train.csv, data/val.csv, data/test.csv
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DISEASE_COLS = ["D", "G", "C", "A", "H", "M"]

# Disease patterns applied per-eye keyword text
DISEASE_PATTERNS = {
    "D": r"diabetic retinopathy|non[\s-]?proliferative retinopathy|proliferative retinopathy|\bnpdr\b|\bpdr\b|diabetic microaneurysm|diabetic fundus",
    "G": r"glaucoma|glaucomatous|suspected glaucoma|cup[\s-]?to[\s-]?disc|c/d ratio|large cup|optic disc cup|pathological cupping",
    "C": r"cataract|lens opacity|cortical cataract|nuclear cataract|subcapsular cataract",
    "A": r"macular degeneration|age[\s-]?related macular degeneration|\bamd\b|\barmd\b|choroidal neovascularization|geographic atrophy",
    "H": r"hypertensive retinopathy|retinal arteriosclerosis|arteriosclerosis|copper wire|silver wire|arteriolar narrowing",
    "M": r"pathological myopia|high myopia|myopic retinopathy|myopic maculopathy|tessellated fundus|myopia",
}

NORMAL_PATTERNS = r"normal fundus|no abnormalities|normal retinal fundus|no obvious abnormalities|normal"

# Fixed DR severity regex patterns (strict non-overlapping matching)
DR_SEVERITY_PATTERNS = [
    (4, r"(?<!non[\s-])(?<!non)(?<!no )proliferative\s+(diabetic\s+)?retinopathy|\bpdr\b"),
    (3, r"severe\s+non[\s-]?proliferative\s+(diabetic\s+)?retinopathy|severe\s+npdr"),
    (2, r"moderate\s+non[\s-]?proliferative\s+(diabetic\s+)?retinopathy|moderate\s+npdr"),
    (1, r"mild\s+non[\s-]?proliferative\s+(diabetic\s+)?retinopathy|mild\s+npdr"),
]


def extract_dr_severity(keyword_text: str) -> int:
    """Rule-based DR severity extraction from eye diagnostic keywords.
    Returns:
      4: Proliferative DR (PDR)
      3: Severe NPDR
      2: Moderate NPDR
      1: Mild NPDR
      0: DR mentioned but grade unspecified
     -1: No DR / Not applicable
    """
    if not isinstance(keyword_text, str) or not keyword_text.strip():
        return -1
    text = keyword_text.lower()

    # Check specific grades first
    for severity, pattern in DR_SEVERITY_PATTERNS:
        if re.search(pattern, text):
            return severity

    # If general DR is mentioned without a specific grade
    if re.search(r"diabetic retinopathy|retinopathy|\bdr\b|\bnpdr\b", text):
        return 0

    return -1


def parse_eye_diagnosis(keyword_text: str, patient_flags: dict = None):
    """Parses disease presence and normal/other status for a single eye.
    Returns:
      labels: dict with keys D, G, C, A, H, M (0 or 1 each)
      dr_severity: int (-1 to 4)
      is_normal: bool
      is_other: bool (has other pathology but no target disease and not normal)
    """
    labels = {d: 0 for d in DISEASE_COLS}
    dr_severity = -1
    is_normal = False
    is_other = False

    if not isinstance(keyword_text, str) or not keyword_text.strip():
        # Fallback if keyword is empty: check patient-level flags
        if patient_flags:
            if patient_flags.get("N", 0) == 1:
                is_normal = True
            else:
                for d in DISEASE_COLS:
                    labels[d] = int(patient_flags.get(d, 0))
                if labels["D"] == 1:
                    dr_severity = 0
        return labels, dr_severity, is_normal, is_other

    text = keyword_text.lower()

    # 1. Check for normal fundus
    if re.search(NORMAL_PATTERNS, text):
        is_normal = True

    # 2. Check each target disease
    for d, pattern in DISEASE_PATTERNS.items():
        if re.search(pattern, text):
            labels[d] = 1
            is_normal = False  # If a disease is detected, it is not normal

    # 3. DR severity extraction
    if labels["D"] == 1:
        dr_severity = extract_dr_severity(text)
    else:
        dr_severity = -1

    # 4. Determine if this eye is purely "Other" (has non-target condition like ERM, RVO, etc.)
    has_target_disease = any(labels[d] == 1 for d in DISEASE_COLS)
    if not has_target_disease and not is_normal:
        is_other = True

    return labels, dr_severity, is_normal, is_other


def build_eye_level_dataset(df: pd.DataFrame, img_dir: str):
    """Processes each patient and extracts independent Left/Right eye records."""
    records = []
    stats = {
        "total_eyes_raw": 0,
        "missing_file_count": 0,
        "normal_eyes": 0,
        "abnormal_eyes": 0,
        "excluded_other_eyes": 0,
    }

    for _, r in df.iterrows():
        patient_id = r["ID"]
        age = r.get("Age", None)
        sex = r.get("Sex", None)

        patient_flags = {
            col: int(r[col]) if col in r and not pd.isna(r[col]) else 0
            for col in ["N", "D", "G", "C", "A", "H", "M", "O"]
        }

        for side in ["Left", "Right"]:
            fname = r.get(f"{side}-Fundus")
            keywords = r.get(f"{side}-Diagnostic Keywords", "")

            if not isinstance(fname, str) or fname.strip() == "":
                continue

            stats["total_eyes_raw"] += 1
            img_path = os.path.join(img_dir, fname)

            # Verify image existence
            if not os.path.isfile(img_path):
                stats["missing_file_count"] += 1
                continue

            labels, dr_sev, is_normal, is_other = parse_eye_diagnosis(keywords, patient_flags)

            if is_other:
                stats["excluded_other_eyes"] += 1
                continue  # Exclude ambiguous "Other" eyes at eye-level

            if is_normal:
                stats["normal_eyes"] += 1
            else:
                stats["abnormal_eyes"] += 1

            row = {
                "patient_id": patient_id,
                "filename": fname,
                "eye": side.lower(),
                "age": age,
                "sex": sex,
                "keywords": keywords,
                "is_normal": int(is_normal),
                "dr_severity": dr_sev,
            }
            for d in DISEASE_COLS:
                row[d] = labels[d]

            records.append(row)

    dataset_df = pd.DataFrame(records)
    return dataset_df, stats


def print_split_summary(df: pd.DataFrame, split_name: str):
    """Prints comprehensive label and severity distribution for a split."""
    print(f"\n=======================================================")
    print(f"               {split_name.upper()} SPLIT SUMMARY")
    print(f"=======================================================")
    print(f"Total eye samples:     {len(df)}")
    print(f"Unique patients:       {df['patient_id'].nunique()}")
    print(f"Normal eye count:      {df['is_normal'].sum()} ({df['is_normal'].mean()*100:.1f}%)")
    print(f"Abnormal eye count:    {(df['is_normal'] == 0).sum()} ({(1-df['is_normal'].mean())*100:.1f}%)")

    print("\nDisease Positive Counts:")
    for d in DISEASE_COLS:
        pos = df[d].sum()
        pct = pos / len(df) * 100 if len(df) > 0 else 0
        print(f"  {d}: {pos:5d} ({pct:5.2f}%)")

    print("\nDR Severity Distribution:")
    sev_counts = df["dr_severity"].value_counts().to_dict()
    for s in [-1, 0, 1, 2, 3, 4]:
        label_desc = {
            -1: "Non-DR (-1)",
            0: "Unspecified DR (0)",
            1: "Mild NPDR (1)",
            2: "Moderate NPDR (2)",
            3: "Severe NPDR (3)",
            4: "Proliferative DR (4)",
        }[s]
        cnt = sev_counts.get(s, 0)
        print(f"  Grade {s:2d} [{label_desc:22s}]: {cnt:5d}")


def patient_wise_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42):
    """Strict patient-wise split preventing contralateral eye leakage."""
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
    ap = argparse.ArgumentParser(description="RetinaSense Eye-Level Data Preprocessing")
    ap.add_argument("--csv_path", default="dataset/ODIR-5K/data.xlsx", help="path to ODIR data.xlsx or full_df.csv")
    ap.add_argument("--img_dir", default="dataset/ODIR-5K/Training Images", help="path to fundus images directory")
    ap.add_argument("--out_dir", default="./data", help="output directory for processed CSVs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("=======================================================")
    print("        RetinaSense Eye-Level Data Preprocessing")
    print("=======================================================")
    print(f"Metadata file:   {args.csv_path}")
    print(f"Images folder:   {args.img_dir}")
    print(f"Output folder:   {args.out_dir}")

    if args.csv_path.endswith(".xlsx") or args.csv_path.endswith(".xls"):
        raw_df = pd.read_excel(args.csv_path)
    else:
        raw_df = pd.read_csv(args.csv_path)

    print(f"\nLoaded {len(raw_df)} patient rows from metadata.")

    # Build eye-level dataset
    eye_df, stats = build_eye_level_dataset(raw_df, args.img_dir)

    print("\n--- Preprocessing Extraction Statistics ---")
    print(f"Total raw eyes evaluated:      {stats['total_eyes_raw']}")
    print(f"Missing image files dropped:   {stats['missing_file_count']}")
    print(f"Excluded 'Other' eyes:         {stats['excluded_other_eyes']}")
    print(f"Preserved Normal eyes:         {stats['normal_eyes']}")
    print(f"Preserved Disease eyes:        {stats['abnormal_eyes']}")
    print(f"Total clean eye dataset size:  {len(eye_df)}")

    # Patient-wise train/val/test split
    train_df, val_df, test_df = patient_wise_split(eye_df)

    # Print summary statistics
    print_split_summary(train_df, "Train")
    print_split_summary(val_df, "Validation")
    print_split_summary(test_df, "Test")

    # Sample Inspection (Print at least 15 generated rows)
    print("\n=======================================================")
    print("        SAMPLE GENERATED ROWS (MANUAL INSPECTION)")
    print("=======================================================")
    cols_to_print = ["filename", "keywords", "D", "G", "C", "A", "H", "M", "dr_severity"]
    sample_df = eye_df[cols_to_print].sample(min(15, len(eye_df)), random_state=42)
    for idx, row in sample_df.iterrows():
        clean_kw = str(row['keywords']).replace('\uff0c', ', ').encode('ascii', 'replace').decode('ascii')
        print(f"File: {row['filename']:15s} | D={row['D']} G={row['G']} C={row['C']} A={row['A']} H={row['H']} M={row['M']} | DR_Sev={row['dr_severity']:2d} | Keywords: {clean_kw}")

    # Save to disk
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False, encoding="utf-8")
    print(f"\n[SUCCESS] Successfully saved train.csv, val.csv, and test.csv to {args.out_dir}\n")


if __name__ == "__main__":
    main()