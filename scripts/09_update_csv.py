"""
Script 09b: Update Labels CSV with Pseudo-Mask Paths
=====================================================
After 09_run_inference.sh has generated pseudo-masks, this script updates
combined_labels_simple.csv (or any source CSV) to fill in the mask_path
column for previously unmasked cases.

This means Script 04 will use mask-guided cropping (crop_around_mask)
for ALL cases instead of falling back to the pelvic-biased centre crop.

Usage:
  python /workspace/scripts/09_update_csv.py

  # Or with custom paths:
  python 09_update_csv.py \\
      --labels_csv /data/classification/combined_labels_simple.csv \\
      --pseudo_mask_dir /data/classification/pseudo_masks \\
      --output_csv /data/classification/combined_labels_with_masks.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ===========================================================================
DEFAULT_LABELS_CSV    = Path("/data/classification/combined_labels_simple.csv")
DEFAULT_PSEUDO_DIR    = Path("/data/classification/pseudo_masks")
DEFAULT_OUTPUT_CSV    = Path("/data/classification/combined_labels_with_masks.csv")
# ===========================================================================


def find_pseudo_mask(patient_id: str, pseudo_mask_dir: Path) -> str | None:
    """
    Look for pseudo mask for a given patient_id.
    Checks:
      <pseudo_mask_dir>/<patient_id>/mask.nii.gz
      <pseudo_mask_dir>/<patient_id_with_underscores>/mask.nii.gz
    """
    # Direct match
    candidate = pseudo_mask_dir / patient_id / "mask.nii.gz"
    if candidate.exists():
        return str(candidate)

    # Hyphen -> underscore (nnU-Net safe_id conversion)
    safe_id = patient_id.replace("-", "_")
    candidate2 = pseudo_mask_dir / safe_id / "mask.nii.gz"
    if candidate2.exists():
        return str(candidate2)

    return None


def main(args):
    log.info("=" * 60)
    log.info("  Script 09b: Update Labels CSV with Pseudo-Mask Paths")
    log.info("=" * 60)

    if not args.labels_csv.exists():
        log.error(f"Labels CSV not found: {args.labels_csv}")
        return
    if not args.pseudo_mask_dir.exists():
        log.error(f"Pseudo-mask directory not found: {args.pseudo_mask_dir}")
        log.error("Run 09_run_inference.sh first.")
        return

    df = pd.read_csv(args.labels_csv)
    log.info(f"Loaded {len(df)} rows from {args.labels_csv}")

    # Check current mask coverage
    has_mask = df["mask_path"].notna() & \
               (df["mask_path"].astype(str).str.strip().isin(["", "nan"]) == False)
    log.info(f"Currently have mask_path: {has_mask.sum()} / {len(df)}")

    # Fill in pseudo-mask paths for cases without masks
    filled = 0
    not_found = 0

    for idx, row in df.iterrows():
        pid = str(row["patient_id"])
        current_mask = str(row.get("mask_path", "")).strip()
        already_has_mask = current_mask not in ("", "nan") and \
                           Path(current_mask).exists()

        if already_has_mask:
            continue  # Ground-truth mask already present — don't overwrite

        pseudo = find_pseudo_mask(pid, args.pseudo_mask_dir)
        if pseudo:
            df.at[idx, "mask_path"] = pseudo
            filled += 1
        else:
            not_found += 1
            log.warning(f"  No pseudo-mask found for {pid}")

    log.info(f"\nFilled {filled} pseudo-mask paths")
    log.info(f"Still no mask: {not_found} cases")

    # Final coverage
    has_mask_after = df["mask_path"].notna() & \
                     (df["mask_path"].astype(str).str.strip().isin(["", "nan"]) == False)
    log.info(f"Total with mask after update: {has_mask_after.sum()} / {len(df)}")

    df.to_csv(args.output_csv, index=False)
    log.info(f"\nUpdated CSV saved -> {args.output_csv}")

    print("\nNext step:")
    print("  python /workspace/scripts/04_train_classifier.py \\")
    print(f"    --labels_csv {args.output_csv} \\")
    print("    --output_dir /data/classification/model_output \\")
    print("    --batch_size 2 --epochs 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv",     type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--pseudo_mask_dir", type=Path, default=DEFAULT_PSEUDO_DIR)
    parser.add_argument("--output_csv",     type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()
    main(args)
