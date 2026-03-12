#!/bin/bash
# =============================================================================
# Script 09: nnU-Net Inference — Generate Pseudo-Masks for Unmasked Cases
# =============================================================================
# Uses the trained nnU-Net model to segment the uterus in CT volumes that
# have no ground-truth masks (TCGA-UCEC, ECPC-IDS without masks, CPTAC-nomask).
#
# These pseudo-masks are then used by Script 04 for mask-guided cropping
# instead of the pelvic-biased fallback crop, improving classifier accuracy.
#
# Workflow:
#   1. Collect all NIfTI images that have no mask
#   2. Prepare them in nnU-Net imagesTs/ format  (Script 09a - Python helper)
#   3. Run nnU-Net predict
#   4. Output masks land in pseudo_masks/<patient_id>/mask.nii.gz
#   5. Update the labels CSV to point mask_path to these pseudo-masks
#
# Usage (on server, inside Docker):
#   bash /workspace/scripts/09_run_inference.sh
#
# Or manually with custom paths:
#   INPUT_DIR=/my/niftis OUTPUT_DIR=/my/masks bash 09_run_inference.sh
# =============================================================================

set -euo pipefail

# ── Paths (all inside Docker mounts) ─────────────────────────────────────────
NNUNET_RESULTS=/data/nnunet_results
DATASET_NAME=Dataset101_EndometrialCancer
TRAINER=nnUNetTrainer
PLANS=nnUNetPlans
CONFIG=3d_fullres
FOLD=0

# Input: directory of NIfTI images to run inference on
# Each file should be named <patient_id>_0000.nii.gz (nnU-Net convention)
# Run 09_prepare_inference.py first to set this up
INFERENCE_INPUT=/data/inference/input
INFERENCE_OUTPUT=/data/inference/output     # predicted masks land here
PSEUDO_MASK_DIR=/data/classification/pseudo_masks

# ── Env vars nnU-Net needs ────────────────────────────────────────────────────
export nnUNet_raw=/data/nnunet_raw
export nnUNet_preprocessed=/data/nnunet_preprocessed
export nnUNet_results=/data/nnunet_results

# ── Check model exists ────────────────────────────────────────────────────────
MODEL_DIR="${NNUNET_RESULTS}/${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"
if [ ! -f "${MODEL_DIR}/checkpoint_best.pth" ]; then
    echo "ERROR: Model checkpoint not found at ${MODEL_DIR}/checkpoint_best.pth"
    echo "       Make sure nnU-Net training (Script 05) has completed first."
    exit 1
fi
echo "Using model: ${MODEL_DIR}"

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "${INFERENCE_INPUT}"
mkdir -p "${INFERENCE_OUTPUT}"
mkdir -p "${PSEUDO_MASK_DIR}"

# ── Check input directory has files ──────────────────────────────────────────
N_INPUT=$(find "${INFERENCE_INPUT}" -name "*_0000.nii.gz" | wc -l)
if [ "${N_INPUT}" -eq 0 ]; then
    echo "ERROR: No input files found in ${INFERENCE_INPUT}"
    echo "       Run 09_prepare_inference.py first to copy and rename NIfTIs."
    exit 1
fi
echo "Found ${N_INPUT} images for inference"

# ── Run nnU-Net inference ─────────────────────────────────────────────────────
echo ""
echo "Running nnU-Net inference..."
echo "  Input:  ${INFERENCE_INPUT}"
echo "  Output: ${INFERENCE_OUTPUT}"
echo ""

nnUNetv2_predict \
    -i  "${INFERENCE_INPUT}" \
    -o  "${INFERENCE_OUTPUT}" \
    -d  101 \
    -c  "${CONFIG}" \
    -f  "${FOLD}" \
    --save_probabilities \
    -chk checkpoint_best.pth

echo ""
echo "Inference complete."

# ── Reorganise predicted masks into pseudo_masks/<patient_id>/mask.nii.gz ─────
echo "Reorganising predicted masks..."
python3 - << 'PYEOF'
import shutil
from pathlib import Path

inference_out = Path("/data/inference/output")
pseudo_mask_dir = Path("/data/classification/pseudo_masks")
pseudo_mask_dir.mkdir(parents=True, exist_ok=True)

masks = list(inference_out.glob("*.nii.gz"))
print(f"  Found {len(masks)} predicted masks")

for mask_path in masks:
    # nnU-Net names output files same as input: <patient_id>_0000.nii.gz
    # or sometimes <patient_id>.nii.gz depending on version
    stem = mask_path.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
    patient_dir = pseudo_mask_dir / stem
    patient_dir.mkdir(exist_ok=True)
    dest = patient_dir / "mask.nii.gz"
    shutil.copy2(mask_path, dest)
    print(f"  {stem} -> {dest}")

print(f"\nAll masks saved to {pseudo_mask_dir}")
print("Next: run 09_update_csv.py to add mask_path to your labels CSV")
PYEOF

echo ""
echo "Done. Pseudo-masks are in: ${PSEUDO_MASK_DIR}"
echo ""
echo "Next steps:"
echo "  1. python /workspace/scripts/09_update_csv.py"
echo "     (adds pseudo mask_path to combined_labels_simple.csv)"
echo "  2. Re-run Script 04 with the updated CSV for mask-guided cropping"
