"""
Script 09a: Prepare Inference Input for nnU-Net
================================================
Collects all NIfTI images that have NO ground-truth mask and copies them
into the nnU-Net inference input format: <patient_id>_0000.nii.gz

Run this before 09_run_inference.sh.

Sources scanned for no-mask images:
  - TCGA-UCEC  (classification/tcga_ucec_final.csv  — no mask_path column)
  - ECPC-IDS   (nifti_output/CT/ECPC_*/CT.nii.gz    — no clinical labels)
  - CPTAC-nomask (classification/cptac_nomask/labels_simple.csv — if exists)

Usage:
  # On server (directly):
  python /workspace/scripts/09_prepare_inference.py

  # On server (inside Docker):
  docker run ... endometrial_pipeline:latest \\
      python /workspace/scripts/09_prepare_inference.py
"""

import logging
import shutil
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
#  PATHS
# ===========================================================================
# ── Server paths (used when running inside Docker on almond) ──────────────
NIFTI_OUTPUT_DIR   = Path("/data/nifti_output/CT")       # all NIfTIs
INFERENCE_INPUT    = Path("/data/inference/input")        # nnU-Net expects files here
CLASSIFICATION_DIR = Path("/data/classification")

TCGA_CSV           = CLASSIFICATION_DIR / "tcga_ucec_final.csv"
CPTAC_NOMASK_CSV   = CLASSIFICATION_DIR / "cptac_nomask" / "labels_simple.csv"
CPTAC_MASK_CSV     = CLASSIFICATION_DIR / "cptac_mask" / "labels_simple.csv"
# ===========================================================================


def collect_no_mask_cases() -> list[dict]:
    """
    Returns list of dicts: {patient_id, nifti_path}
    for all cases that have no ground-truth mask.
    """
    cases = []
    seen_ids = set()

    # ── 1. TCGA cases (colleague's CSV) ──────────────────────────────────
    if TCGA_CSV.exists():
        df = pd.read_csv(TCGA_CSV)
        log.info(f"TCGA CSV: {len(df)} rows")
        for _, row in df.iterrows():
            pid = str(row["patient_id"])
            nifti = str(row.get("nifti_path", "")).strip()
            # TCGA has no masks — always include
            if nifti and Path(nifti).exists() and pid not in seen_ids:
                cases.append({"patient_id": pid, "nifti_path": nifti})
                seen_ids.add(pid)
        log.info(f"  Added {len(cases)} TCGA cases")
    else:
        log.warning(f"TCGA CSV not found: {TCGA_CSV}")

    # ── 2. ECPC-IDS cases (no clinical labels, just raw NIfTIs) ──────────
    ecpc_count = 0
    if NIFTI_OUTPUT_DIR.exists():
        for patient_dir in sorted(NIFTI_OUTPUT_DIR.glob("ECPC_*")):
            pid = patient_dir.name
            if pid in seen_ids:
                continue
            ct = patient_dir / "CT.nii.gz"
            mask = patient_dir / "mask.nii.gz"
            # Only add ECPC cases that do NOT have a ground-truth mask
            # (they have masks from DICOM RTStruct, so skip those)
            # If mask.nii.gz exists it was converted from DICOM — skip
            if ct.exists() and not mask.exists():
                cases.append({"patient_id": pid, "nifti_path": str(ct)})
                seen_ids.add(pid)
                ecpc_count += 1
        log.info(f"  Added {ecpc_count} ECPC-IDS cases without masks")
    else:
        log.warning(f"NIfTI output dir not found: {NIFTI_OUTPUT_DIR}")

    # ── 3. CPTAC-nomask cases (friend's CSV — no mask_path) ──────────────
    if CPTAC_NOMASK_CSV.exists():
        df = pd.read_csv(CPTAC_NOMASK_CSV)
        before = len(cases)
        for _, row in df.iterrows():
            pid = str(row["patient_id"])
            nifti = str(row.get("nifti_path", "")).strip()
            mask = str(row.get("mask_path", "")).strip()
            has_mask = mask not in ("", "nan") and Path(mask).exists()
            if not has_mask and nifti and Path(nifti).exists() and pid not in seen_ids:
                cases.append({"patient_id": pid, "nifti_path": nifti})
                seen_ids.add(pid)
        log.info(f"  Added {len(cases) - before} CPTAC-nomask cases")
    else:
        log.info(f"CPTAC-nomask CSV not found ({CPTAC_NOMASK_CSV}) — skipping")

    # ── 4. Skip CPTAC-masked cases (they already have ground-truth masks) ─
    if CPTAC_MASK_CSV.exists():
        df_mask = pd.read_csv(CPTAC_MASK_CSV)
        masked_ids = set(df_mask["patient_id"].astype(str).tolist())
        before = len(cases)
        cases = [c for c in cases if c["patient_id"] not in masked_ids]
        removed = before - len(cases)
        if removed > 0:
            log.info(f"  Removed {removed} cases that already have ground-truth masks")

    return cases


def main():
    log.info("=" * 60)
    log.info("  Script 09a: Prepare nnU-Net Inference Input")
    log.info("=" * 60)

    INFERENCE_INPUT.mkdir(parents=True, exist_ok=True)

    cases = collect_no_mask_cases()
    log.info(f"\nTotal no-mask cases to process: {len(cases)}")

    if len(cases) == 0:
        log.warning("No cases found. Check your paths.")
        return

    # Copy files into inference input directory with nnU-Net naming
    copied = 0
    skipped = 0
    for case in cases:
        pid = case["patient_id"]
        src = Path(case["nifti_path"])
        if not src.exists():
            log.warning(f"  MISSING: {src}")
            skipped += 1
            continue

        # nnU-Net inference input format: <case_id>_0000.nii.gz
        # Use patient_id directly (replace hyphens with underscores for safety)
        safe_id = pid.replace("-", "_")
        dst = INFERENCE_INPUT / f"{safe_id}_0000.nii.gz"

        if dst.exists():
            log.info(f"  Already exists, skipping: {dst.name}")
            skipped += 1
            continue

        shutil.copy2(src, dst)
        log.info(f"  Copied: {pid} -> {dst.name}")
        copied += 1

    log.info(f"\nDone: {copied} copied, {skipped} skipped")
    log.info(f"Inference input ready at: {INFERENCE_INPUT}")
    log.info(f"Files: {len(list(INFERENCE_INPUT.glob('*_0000.nii.gz')))}")

    print("\nNext step:")
    print("  bash /workspace/scripts/09_run_inference.sh")


if __name__ == "__main__":
    main()
