"""
Script 08: Combine Classification Labels from All Sources
==========================================================
Merges the three separate label CSVs into one combined_labels_simple.csv
ready for Script 04 (3D DenseNet-121 classifier).

Three sources:
  1. CPTAC-masked   — classification/cptac_mask/labels_simple.csv
                      (61 patients, have segmentation masks, mask-guided crop)
  2. CPTAC-nomask   — classification/cptac_nomask/labels_simple.csv
                      (friend's work, full 80 GB CPTAC, pelvic-biased crop)
  3. TCGA-UCEC      — classification/tcga_ucec_final.csv
                      (colleague's work, 62 patients, pelvic-biased crop)

Each source CSV must have these columns (Script 04 minimum requirements):
  patient_id, grade, grade_int, nifti_path, split, source
  mask_path  (optional — leave empty/NaN for nomask sources)

Output:
  classification/combined_labels_simple.csv   (all sources merged)
  classification/combined_labels_summary.txt  (counts per source/grade/split)

Usage:
  # On Windows (local):
  python scripts/08_combine_labels.py

  # On server (inside Docker or directly):
  python /workspace/scripts/08_combine_labels.py \\
      --cptac_mask  /data/classification/cptac_mask/labels_simple.csv \\
      --cptac_nomask /data/classification/cptac_nomask/labels_simple.csv \\
      --tcga        /data/classification/tcga_ucec_final.csv \\
      --output_dir  /data/classification
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("08_combine_labels.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ===========================================================================
#  HARDCODED PATHS (Windows local defaults — override with CLI args on server)
# ===========================================================================
DEFAULT_CPTAC_MASK   = Path(r"F:\endometrial_cancer\classification\cptac_mask\labels_simple.csv")
DEFAULT_CPTAC_NOMASK = Path(r"F:\endometrial_cancer\classification\cptac_nomask\labels_simple.csv")
DEFAULT_TCGA         = Path(r"F:\endometrial_cancer\classification\tcga_ucec_final.csv")
DEFAULT_OUTPUT_DIR   = Path(r"F:\endometrial_cancer\classification")
# ===========================================================================

# Columns Script 04 requires
REQUIRED_COLS = ["patient_id", "grade", "grade_int", "nifti_path", "split"]
# Columns we carry through (optional ones filled with NaN if missing)
KEEP_COLS = ["patient_id", "grade", "grade_int", "nifti_path", "mask_path",
             "split", "source", "histology", "figo_stage", "true_modality",
             "series_category"]

GRADE_MAP = {
    "G1": "G1", "G2": "G2", "G3": "G3", "G4": "G3",
    "High Grade": "G3", "Unknown": None, "Not Reported": None, "'--": None,
}
GRADE_TO_INT = {"G1": 0, "G2": 1, "G3": 2}


def load_and_normalise(path: Path, source_name: str) -> pd.DataFrame | None:
    """
    Load a labels CSV, verify required columns exist, normalise grade columns,
    tag source, and return a standardised DataFrame.
    """
    if not path.exists():
        log.warning(f"  [{source_name}] File not found: {path} — SKIPPING")
        return None

    df = pd.read_csv(path, low_memory=False)
    log.info(f"  [{source_name}] Loaded {len(df)} rows from {path}")
    log.info(f"    Columns: {list(df.columns)}")

    # ── Normalise grade columns ──────────────────────────────────────────
    if "grade" not in df.columns and "diagnoses.tumor_grade" in df.columns:
        df["grade"] = df["diagnoses.tumor_grade"].map(GRADE_MAP)

    if "grade" in df.columns:
        df["grade"] = df["grade"].map(lambda x: GRADE_MAP.get(str(x).strip(), x)
                                      if pd.notna(x) else x)

    if "grade_int" not in df.columns and "grade" in df.columns:
        df["grade_int"] = df["grade"].map(GRADE_TO_INT)

    # ── Normalise nifti_path column ──────────────────────────────────────
    # TCGA uses 'nifti_path'; some older CSVs may use 'image_path' or 'final_output_file'
    if "nifti_path" not in df.columns:
        for alt in ("image_path", "final_output_file", "new_path"):
            if alt in df.columns:
                df["nifti_path"] = df[alt]
                log.info(f"    Renamed '{alt}' -> 'nifti_path'")
                break

    # ── Ensure mask_path column exists ──────────────────────────────────
    if "mask_path" not in df.columns:
        df["mask_path"] = np.nan

    # ── Tag source ──────────────────────────────────────────────────────
    if "source" not in df.columns:
        df["source"] = source_name

    # ── Check required columns ──────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        log.error(f"  [{source_name}] Missing required columns: {missing_cols} — SKIPPING")
        return None

    # ── Filter to valid grades only ──────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["grade_int"]).copy()
    df["grade_int"] = df["grade_int"].astype(int)
    after = len(df)
    if before != after:
        log.info(f"    Dropped {before - after} rows with missing/unknown grade")

    log.info(f"    Final: {len(df)} rows | grade dist: "
             f"{df['grade'].value_counts().to_dict()}")

    # ── Keep only standard columns (fill missing with NaN) ──────────────
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df[KEEP_COLS].copy()


def main(args):
    log.info("=" * 65)
    log.info("  Script 08: Combine Classification Labels")
    log.info("=" * 65)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load each source ─────────────────────────────────────────────────
    frames = []

    log.info("\nLoading CPTAC-masked labels...")
    df_cptac_mask = load_and_normalise(args.cptac_mask, "CPTAC_masked")
    if df_cptac_mask is not None:
        frames.append(df_cptac_mask)

    if args.cptac_nomask.exists():
        log.info("\nLoading CPTAC-nomask labels (friend's data)...")
        df_cptac_nomask = load_and_normalise(args.cptac_nomask, "CPTAC_nomask")
        if df_cptac_nomask is not None:
            frames.append(df_cptac_nomask)
    else:
        log.info(f"\nCPTAC-nomask CSV not found ({args.cptac_nomask}) — skipping")

    if args.tcga.exists():
        log.info("\nLoading TCGA-UCEC labels (colleague's data)...")
        df_tcga = load_and_normalise(args.tcga, "TCGA")
        if df_tcga is not None:
            frames.append(df_tcga)
    else:
        log.info(f"\nTCGA CSV not found ({args.tcga}) — skipping")

    if not frames:
        log.error("No valid label sources found. Exiting.")
        return

    # ── Concatenate ──────────────────────────────────────────────────────
    combined = pd.concat(frames, ignore_index=True)
    log.info(f"\nCombined: {len(combined)} total cases from {len(frames)} source(s)")

    # Check for duplicate patient IDs across sources
    dupes = combined[combined.duplicated(subset=["patient_id"], keep=False)]
    if len(dupes) > 0:
        log.warning(f"  {len(dupes)} rows have duplicate patient_ids across sources:")
        log.warning(f"  {dupes['patient_id'].unique().tolist()}")
        log.warning("  Review and deduplicate manually if needed.")

    # ── Save ─────────────────────────────────────────────────────────────
    out_csv = args.output_dir / "combined_labels_simple.csv"
    combined.to_csv(out_csv, index=False)
    log.info(f"\nCombined labels saved -> {out_csv}")

    # ── Summary report ───────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append("  COMBINED LABEL SUMMARY")
    summary_lines.append("=" * 65)
    summary_lines.append(f"  Total cases: {len(combined)}")
    summary_lines.append("")

    for src in combined["source"].unique():
        sub = combined[combined["source"] == src]
        summary_lines.append(f"  Source: {src}  ({len(sub)} cases)")
        has_mask = sub["mask_path"].notna() & (sub["mask_path"].astype(str).str.strip() != "nan")
        summary_lines.append(f"    With mask:    {has_mask.sum()}")
        summary_lines.append(f"    Without mask: {(~has_mask).sum()}")
        for grade in ["G1", "G2", "G3"]:
            cnt = (sub["grade"] == grade).sum()
            bar = "█" * cnt
            summary_lines.append(f"    {grade}: {cnt:3d}  {bar}")
        summary_lines.append("")

    summary_lines.append("  SPLIT BREAKDOWN:")
    for split in ["train", "val", "test"]:
        sub = combined[combined["split"] == split]
        if len(sub) == 0:
            continue
        summary_lines.append(f"  {split.upper()} ({len(sub)} cases):")
        for grade in ["G1", "G2", "G3"]:
            cnt = (sub["grade"] == grade).sum()
            summary_lines.append(f"    {grade}: {cnt}")
    summary_lines.append("")

    counts = combined["grade_int"].value_counts()
    if len(counts) > 1:
        ratio = counts.max() / counts.min()
        summary_lines.append(f"  Class imbalance ratio: {ratio:.1f}:1")
        if ratio > 3:
            summary_lines.append("  NOTE: Script 04 WeightedRandomSampler handles this.")
    summary_lines.append("")
    summary_lines.append(f"  Output: {out_csv}")
    summary_lines.append("=" * 65)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_out = args.output_dir / "combined_labels_summary.txt"
    summary_out.write_text(summary_text, encoding="utf-8")
    log.info(f"Summary saved -> {summary_out}")

    print("\nNext step — run Script 04 on the server:")
    print("  docker run ... endometrial_pipeline:latest \\")
    print("    python /workspace/scripts/04_train_classifier.py \\")
    print(f"      --labels_csv /data/classification/combined_labels_simple.csv \\")
    print(f"      --output_dir /data/classification/model_output \\")
    print(f"      --batch_size 2 --epochs 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine classification label CSVs")
    parser.add_argument("--cptac_mask",   type=Path, default=DEFAULT_CPTAC_MASK)
    parser.add_argument("--cptac_nomask", type=Path, default=DEFAULT_CPTAC_NOMASK)
    parser.add_argument("--tcga",         type=Path, default=DEFAULT_TCGA)
    parser.add_argument("--output_dir",   type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    main(args)
