"""
Script 03 (v4): CPTAC-UCEC Clinical Metadata -> Classification Labels
======================================================================
Extracts histological grade (G1/G2/G3) from the GDC clinical TSV
and aligns with nnU-Net case IDs from Script 02's case_split_log.csv.

OUTPUT: labels for the CPTAC *masked* subset only.
  → saved to classification/cptac_mask/labels_simple.csv
  → nifti_path and mask_path point to server Linux paths under
    /mnt/data/shared-data/endometrial-cancer/nifti_output/CT/

The pipeline has three separate label sources:
  1. cptac_mask/   — this script (CPTAC cases that have segmentation masks)
  2. cptac_nomask/ — handled by friend (full 80 GB CPTAC, no masks)
  3. tcga_images/  — handled by colleague (TCGA, no masks, tcga_ucec_final.csv)

To combine all three for Script 04, concatenate the CSVs:
  pd.concat([cptac_mask_df, cptac_nomask_df, tcga_df]).to_csv("combined_labels_simple.csv")
  Each must have: patient_id, grade, grade_int, nifti_path, mask_path, split, source

CONFIRMED COLUMN NAMES (from actual clinical.tsv inspection):
  Patient ID : cases.submitter_id
  Grade      : diagnoses.tumor_grade  (values: G1, G2, G3, G4, Unknown, High Grade)
  Histology  : diagnoses.primary_diagnosis
  Site filter: cases.primary_site     (filter: contains 'Uter')
  Project    : project.project_id     (= 'CPTAC-3', not 'CPTAC-UCEC')

NOTES from data analysis:
  - The TSV is project CPTAC-3 (multi-cancer GDC export), NOT CPTAC-UCEC only.
    Must filter by cases.primary_site = 'Uterus, NOS'
  - 241 uterine patients total, 63 match our imaging cohort
  - All 63 have clean G1/G2/G3 labels (no missing grades in imaging cohort)
  - 4 imaging patients missing from clinical: C3L-00962, C3N-01001,
    C3N-01763, C3N-03044 -> will be excluded from classifier training
  - G4 -> mapped to G3 (undifferentiated/poorly differentiated)
  - Multiple rows per patient (treatments etc) -> deduplicated on first occurrence

Requirements:
    pip install pandas numpy

Usage:
    python 03_prepare_classification_labels.py
    (no arguments needed - all paths hardcoded below)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("03_prepare_labels.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ===========================================================================
#  EDIT THESE PATHS
# ===========================================================================
CLINICAL_TSV  = Path(r"F:\endometrial_cancer\data\clinical.tsv")

# Output of Script 02 — used to map patient_id -> nnunet_id + nifti/mask paths
SPLIT_LOG_CSV = Path(r"F:\endometrial_cancer\data\nnunet_raw\Dataset101_EndometrialCancer\case_split_log.csv")

# Where to write label files (read by Script 04)
# cptac_mask/ = CPTAC cases that have segmentation masks (this script's scope)
OUTPUT_DIR    = Path(r"F:\endometrial_cancer\classification\cptac_mask")

# Server path prefix — Script 02 writes Windows paths into case_split_log.csv.
# These are rewritten to Linux server paths in the output CSV so Script 04
# can run directly on the server without any further path fixing.
# Set to None to keep the original Windows paths (e.g. for local testing).
WINDOWS_NIFTI_PREFIX = r"F:\nifti_output"
SERVER_NIFTI_PREFIX  = "/mnt/data/shared-data/endometrial-cancer/nifti_output"
# ===========================================================================

# Grade mapping: raw GDC string -> canonical label
GRADE_MAP = {
    "G1":         "G1",
    "G2":         "G2",
    "G3":         "G3",
    "G4":         "G3",   # Undifferentiated -> treat as G3
    "High Grade": "G3",
    "Unknown":    None,
    "Not Reported": None,
    "'--":        None,
}
GRADE_TO_INT = {"G1": 0, "G2": 1, "G3": 2}


def load_clinical(tsv_path: Path) -> pd.DataFrame:
    """
    Load and filter the GDC clinical TSV to uterine cancer patients only.
    Deduplicates so there is exactly one row per patient.
    Returns DataFrame with columns: patient_id, raw_grade, grade, grade_int
    """
    log.info(f"Loading clinical data from: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    log.info(f"  Total rows: {len(df)} | Unique patients: {df['cases.submitter_id'].nunique()}")
    log.info(f"  Projects in file: {df['project.project_id'].unique().tolist()}")

    # Filter to uterine/endometrial cancer patients
    uterus_mask = df["cases.primary_site"].str.contains("Uter", case=False, na=False)
    df_uter = df[uterus_mask].copy()
    log.info(f"  Uterine patients: {df_uter['cases.submitter_id'].nunique()} unique patients "
             f"({len(df_uter)} rows before dedup)")

    # Deduplicate: one row per patient (multiple rows arise from multiple treatments)
    df_dedup = df_uter.drop_duplicates(subset="cases.submitter_id", keep="first")
    log.info(f"  After dedup: {len(df_dedup)} patients")

    # Extract and map grade
    df_dedup = df_dedup.copy()
    df_dedup["patient_id"] = df_dedup["cases.submitter_id"].astype(str).str.strip()
    df_dedup["raw_grade"]  = df_dedup["diagnoses.tumor_grade"].astype(str).str.strip()
    df_dedup["grade"]      = df_dedup["raw_grade"].map(GRADE_MAP)
    df_dedup["grade_int"]  = df_dedup["grade"].map(GRADE_TO_INT)

    # Log raw grade distribution
    log.info(f"\n  Raw grade distribution:")
    for val, cnt in df_dedup["raw_grade"].value_counts(dropna=False).items():
        mapped = GRADE_MAP.get(val, "UNMAPPED")
        log.info(f"    '{val}': {cnt} -> {mapped}")

    result = df_dedup[["patient_id", "raw_grade", "grade", "grade_int",
                        "diagnoses.primary_diagnosis", "diagnoses.figo_stage"]].copy()
    result.columns = ["patient_id", "raw_grade", "grade", "grade_int",
                      "histology", "figo_stage"]
    return result


def rewrite_path(p: str) -> str:
    """
    Rewrite a Windows nifti/mask path from case_split_log.csv to its
    equivalent Linux server path.  Handles both forward and backslashes.
    If WINDOWS_NIFTI_PREFIX is None, returns the path unchanged.
    """
    if WINDOWS_NIFTI_PREFIX is None:
        return p
    if not isinstance(p, str) or p.strip() in ("", "nan"):
        return p
    # Normalise backslashes to forward slashes for comparison
    p_norm = p.replace("\\", "/")
    win_norm = WINDOWS_NIFTI_PREFIX.replace("\\", "/")
    if p_norm.startswith(win_norm):
        return SERVER_NIFTI_PREFIX + p_norm[len(win_norm):]
    return p


def reassign_splits(df: pd.DataFrame, val_fraction: float = 0.15) -> pd.DataFrame:
    """
    Ensures train/val/test splits with strict patient-level integrity:
      - No patient appears in more than one split
      - Val is carved from the existing train set (Script 02 only produces train/test)
      - Stratified by grade so each split has a balanced grade distribution
      - test set from Script 02 is kept as-is

    Args:
        df:           DataFrame with columns patient_id, grade, split
        val_fraction: fraction of train *patients* to move to val (default 15%)
    Returns:
        DataFrame with updated 'split' column (train / val / test)
    """
    from sklearn.model_selection import train_test_split

    # ── Verify patient-level integrity of existing split ────────────────
    # Each patient should appear in only one original split
    patient_splits = df.groupby("patient_id")["split"].nunique()
    leaked = patient_splits[patient_splits > 1]
    if len(leaked) > 0:
        log.warning(f"  {len(leaked)} patients appear in multiple splits (leak): "
                    f"{leaked.index.tolist()}")
        log.warning("  Keeping first-seen split for these patients.")
        first_split = df.drop_duplicates(subset="patient_id", keep="first")[
            ["patient_id", "split"]
        ].set_index("patient_id")["split"]
        df = df.copy()
        df["split"] = df["patient_id"].map(first_split)

    # ── Identify train patients ──────────────────────────────────────────
    train_patients = (
        df[df["split"] == "train"][["patient_id", "grade"]]
        .drop_duplicates("patient_id")
        .reset_index(drop=True)
    )
    log.info(f"\n  Train patients before val carve: {len(train_patients)}")

    if len(train_patients) < 6:
        log.warning("  Too few train patients to carve val — keeping train/test only.")
        return df

    # ── Stratified split at patient level ────────────────────────────────
    try:
        tr_pids, va_pids = train_test_split(
            train_patients["patient_id"],
            test_size=val_fraction,
            stratify=train_patients["grade"],
            random_state=42,
        )
    except ValueError:
        log.warning("  Stratified val split failed (too few per class) — using random split.")
        tr_pids, va_pids = train_test_split(
            train_patients["patient_id"],
            test_size=val_fraction,
            random_state=42,
        )

    # ── Reassign splits ──────────────────────────────────────────────────
    df = df.copy()
    df.loc[df["patient_id"].isin(va_pids), "split"] = "val"
    df.loc[df["patient_id"].isin(tr_pids), "split"] = "train"
    # test remains unchanged

    # ── Log result ───────────────────────────────────────────────────────
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        pts = sub["patient_id"].nunique()
        grade_dist = sub["grade"].value_counts().to_dict()
        log.info(f"  {s:5s}: {pts} patients | {grade_dist}")

    # ── Final integrity check ────────────────────────────────────────────
    all_splits = df.groupby("patient_id")["split"].nunique()
    leaked_after = all_splits[all_splits > 1]
    if len(leaked_after) > 0:
        log.error(f"  PATIENT LEAK AFTER SPLIT: {leaked_after.index.tolist()}")
    else:
        log.info("  Patient-level integrity check: PASSED (no leaks)")

    return df


def merge_with_split_log(labels_df: pd.DataFrame, split_log_path: Path) -> pd.DataFrame:
    """
    Join grade labels onto the nnU-Net split log using patient_id.
    Split log columns: nnunet_id, patient_id, patient_key, folder_modality,
                       true_modality, image_path/nifti_path, mask_path,
                       mask_voxels, split
    Rewrites nifti_path and mask_path from Windows to server Linux paths.
    Adds source='CPTAC' column for downstream merging with TCGA/ECPC CSVs.
    """
    split_df = pd.read_csv(split_log_path)
    log.info(f"\nSplit log: {len(split_df)} cases, columns: {list(split_df.columns)}")

    # Normalize column name: Script 02 v4 used 'image_path', v5 added 'nifti_path' alias
    if "nifti_path" not in split_df.columns and "image_path" in split_df.columns:
        split_df["nifti_path"] = split_df["image_path"]

    merged = split_df.merge(
        labels_df[["patient_id", "grade", "grade_int", "raw_grade",
                   "histology", "figo_stage"]],
        on="patient_id",
        how="left"
    )

    has_grade = merged["grade"].notna()
    log.info(f"\n  Imaging patients with grade:    {has_grade.sum()} / {len(merged)}")
    log.info(f"  Imaging patients WITHOUT grade: {(~has_grade).sum()}")
    if (~has_grade).any():
        missing = merged[~has_grade]["patient_id"].tolist()
        log.warning(f"  Missing grade labels for: {missing}")
        log.warning("  These patients will be EXCLUDED from classifier training")

    # Rewrite paths from Windows to Linux server
    if WINDOWS_NIFTI_PREFIX is not None:
        log.info(f"\n  Rewriting paths: {WINDOWS_NIFTI_PREFIX!r} -> {SERVER_NIFTI_PREFIX!r}")
        merged["nifti_path"] = merged["nifti_path"].astype(str).apply(rewrite_path)
        if "mask_path" in merged.columns:
            merged["mask_path"] = merged["mask_path"].astype(str).apply(rewrite_path)

    # Tag source dataset for downstream merging
    merged["source"] = "CPTAC"

    return merged


def main():
    log.info("=" * 65)
    log.info("  CPTAC-UCEC Label Preparation (v3)")
    log.info("=" * 65)

    if not CLINICAL_TSV.exists():
        log.error(f"Clinical TSV not found: {CLINICAL_TSV}")
        return
    if not SPLIT_LOG_CSV.exists():
        log.error(f"Split log not found: {SPLIT_LOG_CSV}")
        log.error("Run Script 02 first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and filter clinical data
    labels_df = load_clinical(CLINICAL_TSV)

    labeled = labels_df.dropna(subset=["grade_int"])
    log.info(f"\nTotal uterine patients with valid grade: {len(labeled)}")
    log.info(f"Grade distribution:\n{labeled['grade'].value_counts().to_string()}")

    # Warn about any unmapped grade strings
    unmapped = labels_df[labels_df["grade"].isna() & (labels_df["raw_grade"] != "None")]
    if len(unmapped) > 0:
        log.warning(f"\nUnmapped grade strings (add to GRADE_MAP if needed):")
        log.warning(unmapped["raw_grade"].value_counts().to_string())

    # Merge with nnU-Net split log
    log.info("\nMerging with nnU-Net split log...")
    final_df = merge_with_split_log(labeled, SPLIT_LOG_CSV)

    # Carve val from train + verify patient-level integrity
    log.info("\nReassigning splits (carving val from train)...")
    final_df = reassign_splits(final_df, val_fraction=0.15)

    # Validate grade_int is only 0/1/2
    valid_ints = {0, 1, 2}
    bad = final_df.dropna(subset=["grade_int"])
    bad = bad[~bad["grade_int"].isin(valid_ints)]
    if len(bad) > 0:
        log.error(f"Invalid grade_int values: {bad[['patient_id','grade_int']].to_string()}")

    # Save full merged file
    full_out = OUTPUT_DIR / "classification_labels.csv"
    final_df.to_csv(full_out, index=False)
    log.info(f"\nFull labels saved -> {full_out}")

    # Save simplified file for Script 04
    # Columns Script 04 needs: nnunet_id, patient_id, grade_int, grade,
    #                           split, nifti_path, mask_path, true_modality, source
    keep_cols = ["nnunet_id", "patient_id", "grade_int", "grade",
                 "split", "nifti_path", "mask_path", "true_modality",
                 "histology", "figo_stage", "source"]
    keep_cols = [c for c in keep_cols if c in final_df.columns]

    simple = final_df[keep_cols].dropna(subset=["grade_int"]).copy()
    simple["grade_int"] = simple["grade_int"].astype(int)
    simple_out = OUTPUT_DIR / "labels_simple.csv"
    simple.to_csv(simple_out, index=False)
    log.info(f"Simple labels saved  -> {simple_out}")
    log.info(f"  Paths rewritten to server: {WINDOWS_NIFTI_PREFIX is not None}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  LABEL SUMMARY")
    print("=" * 65)
    print(f"  Total imaging cases:       {len(final_df)}")
    print(f"  Cases with grade label:    {simple['grade_int'].notna().sum()}")
    print(f"  Cases WITHOUT grade:       {final_df['grade'].isna().sum()}")
    print()

    for split_name in ["train", "val", "test"]:
        sub = simple[simple["split"] == split_name]
        if len(sub) == 0:
            continue
        print(f"  {split_name.upper()} ({len(sub)} cases):")
        for grade, cnt in sub["grade"].value_counts().sort_index().items():
            bar = "█" * cnt
            print(f"    {grade}: {cnt:2d}  {bar}")
        print()

    counts = simple["grade_int"].value_counts()
    if len(counts) > 1:
        ratio = counts.max() / counts.min()
        print(f"  Class imbalance ratio: {ratio:.1f}:1")
        if ratio > 3:
            print("  WARNING: High imbalance — Script 04 WeightedRandomSampler handles this.")
    print()
    print(f"  Output directory: {OUTPUT_DIR}  (cptac_mask/)")
    print("=" * 65)
    print()
    print("Next steps:")
    print(f"  1. Transfer to server:")
    print(f"       scp -r \"{OUTPUT_DIR}\" aavash@almond:/mnt/data/shared-data/endometrial-cancer/classification/")
    print(f"  2. To combine with TCGA/ECPC for Script 04:")
    print(f"       python scripts\\combine_labels.py  (or manually concatenate CSVs)")
    print(f"  3. Run Script 04:")
    print(f"       python scripts\\04_train_classifier.py \\")
    print(f"         --labels_csv /mnt/data/.../classification/cptac_mask/labels_simple.csv \\")
    print(f"         --output_dir /mnt/data/.../classification/model_output")


if __name__ == "__main__":
    main()