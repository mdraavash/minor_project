
set -e

# ─── CONFIG ───────────────────────────────────────────────────────────────
DATASET_ID=101
DATASET_NAME="EndometrialCancer"
FOLD=0
CONFIG="3d_fullres"
N_WORKERS=4

# Paths inside container — match Docker volume mounts above
export nnUNet_raw="/data/nnunet_raw"
export nnUNet_preprocessed="/data/nnunet_preprocessed"
export nnUNet_results="/data/nnunet_results"
# ──────────────────────────────────────────────────────────────────────────

echo "=============================================="
echo " nnU-Net Preprocess + Train (FRESH START)"
echo " Dataset : $DATASET_ID ($DATASET_NAME)"
echo " Config  : $CONFIG"
echo " Fold    : $FOLD"
echo " Workers : $N_WORKERS"
echo "=============================================="

echo ""
echo "[1/4] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "[2/4] Planning and preprocessing..."
nnUNetv2_plan_and_preprocess \
    -d $DATASET_ID \
    --verify_dataset_integrity \
    -c $CONFIG \
    --clean \
    -np $N_WORKERS

echo ""
echo "[3/4] Starting training (fold $FOLD) — fresh start, no checkpoint..."
echo "      Estimated time: ~4 days on RTX 3060 Ti"
echo "      Monitor progress:"
echo "        tail -f $nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}/training_log*.txt"
echo ""

# Fresh start — NO --c flag
# To RESUME from checkpoint add: --c
nnUNetv2_train \
    $DATASET_ID \
    $CONFIG \
    $FOLD \
    --npz \
    -num_gpus 1

echo ""
echo "[4/4] Finding best configuration..."
nnUNetv2_find_best_configuration $DATASET_ID -c $CONFIG

echo ""
echo "=============================================="
echo " Training complete!"
echo " Results: $nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/"
echo "=============================================="
echo ""
echo "To run inference on new images:"
echo "  nnUNetv2_predict \\"
echo "    -i /data/new_images \\"
echo "    -o /data/predictions \\"
echo "    -d $DATASET_ID -c $CONFIG -f $FOLD"