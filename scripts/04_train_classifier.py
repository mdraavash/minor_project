"""
Script 04: 3D DenseNet-121 Endometrial Cancer Grader (G1 / G2 / G3)
=====================================================================
Trains a 3D DenseNet-121 to classify endometrial cancer grade from CT volumes.
Supports CPTAC-UCEC, TCGA-UCEC, or a combined cohort.

Pipeline
--------
1.  Read labels_simple.csv  (or combined_labels_simple.csv)
2.  Load NIfTI volume + mask (mask optional for TCGA without pseudo-masks)
3.  Crop: 96×96×96 voxel bounding box around uterus mask
    If mask is missing  → pelvic-biased crop (Z centre at 60% from top,
                          i.e. bottom 40% of volume, where uterus sits)
4.  Z-score normalise the crop
5.  Data augmentation (training only):
      random flips, 90° rotations, zoom ±20%, Gaussian noise,
      brightness/contrast jitter, Gaussian blur, gamma correction
6.  Two-phase training:
      Phase 1 (5 epochs)  – train classification head only
      Phase 2 (N epochs)  – fine-tune full network + cosine LR + early stopping
7.  WeightedRandomSampler to handle G2-dominant class imbalance
8.  Save best checkpoint, training history CSV, and evaluation plots

Outputs (in --output_dir)
--------------------------
  best_classifier.pth        best model weights
  training_history.csv       loss / accuracy per epoch
  config.json                full run configuration
  tensorboard/               TensorBoard event files
  plots/
    loss_accuracy_curves.png
    confusion_matrix.png
    roc_curves.png
    per_class_metrics.png

Requirements
------------
  pip install torch torchvision monai nibabel pandas scikit-learn \
              matplotlib seaborn tensorboard tqdm

Usage
-----
  # Minimal
  python 04_train_classifier.py \
      --labels_csv /data/classification/labels_simple.csv \
      --output_dir /data/classification/model_output

  # Full options
  python 04_train_classifier.py \
      --labels_csv /data/classification/combined_labels_simple.csv \
      --output_dir /data/classification/model_output_combined \
      --crop_size  96 \
      --crop_margin 20 \
      --batch_size 4 \
      --epochs 50 \
      --lr 1e-4 \
      --patience 15 \
      --seed 42
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import nibabel as nib
from monai.networks.nets import DenseNet121
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)

CLASS_NAMES = ["G1", "G2", "G3"]
NUM_CLASSES = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
#  Volume I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file and return the data array as float32."""
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


def crop_around_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    crop_size: int = 96,
    margin: int = 20,
) -> np.ndarray:
    """
    Extract a cubic region around the bounding box of the mask.
    Pads with the volume's minimum value if the crop extends beyond borders.

    Args:
        volume:    3-D float array  (X, Y, Z)
        mask:      3-D binary array (X, Y, Z)
        crop_size: output cube side length in voxels
        margin:    extra voxels added around the tight bounding box
    Returns:
        Cropped volume of shape (crop_size, crop_size, crop_size)
    """
    fg = np.argwhere(mask > 0)
    if len(fg) == 0:
        # Empty mask — fall back to centre crop
        return centre_crop(volume, crop_size)

    mins = fg.min(axis=0)
    maxs = fg.max(axis=0)
    centre = ((mins + maxs) / 2).astype(int)

    half = crop_size // 2
    pad_val = float(volume.min())

    # Pad volume so we can always extract a full cube
    pad = half + margin
    vol_padded = np.pad(
        volume,
        pad_width=((pad, pad), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=pad_val,
    )
    c = centre + pad  # adjust centre for padding offset

    crop = vol_padded[
        c[0] - half : c[0] + half,
        c[1] - half : c[1] + half,
        c[2] - half : c[2] + half,
    ]
    # Ensure exactly crop_size (handles edge case from integer division)
    crop = crop[:crop_size, :crop_size, :crop_size]
    return crop


def centre_crop(volume: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Extract a cube biased toward the lower pelvis (bottom 40% of the volume
    along the Z/axial axis), where the uterus typically resides.
    Used as fallback when no mask is available (TCGA, ECPC-IDS without masks).

    The XY centre is kept at the image midpoint (uterus is roughly central
    in the axial plane). Only Z is shifted downward.

    Args:
        volume:    3-D float array (X, Y, Z) — axial slices along last axis
        crop_size: output cube side length in voxels
    Returns:
        Cropped volume of shape (crop_size, crop_size, crop_size)
    """
    pad_val = float(volume.min())
    half = crop_size // 2
    pad  = half

    vol_padded = np.pad(
        volume,
        pad_width=((pad, pad), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=pad_val,
    )

    shape = np.array(vol_padded.shape)

    # XY: stay at image centre
    cx = shape[0] // 2
    cy = shape[1] // 2

    # Z: shift to 40% from the bottom of the original (unpadded) volume.
    # In NIfTI convention superior slices have higher index, so
    # "bottom of pelvis" = lower Z index = ~60% from top = 40% from bottom.
    # We place the crop centre at 60% of the original depth (from index 0).
    orig_z = volume.shape[2]
    z_centre_orig = int(orig_z * 0.60)   # 60% from top = 40% from bottom
    cz = z_centre_orig + pad             # adjust for padding offset

    crop = vol_padded[
        cx - half : cx + half,
        cy - half : cy + half,
        cz - half : cz + half,
    ]
    return crop[:crop_size, :crop_size, :crop_size]


def zscore_normalise(volume: np.ndarray) -> np.ndarray:
    """Z-score normalise a volume, avoiding division by zero."""
    mean = volume.mean()
    std  = volume.std()
    if std < 1e-6:
        return volume - mean
    return (volume - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation
# ─────────────────────────────────────────────────────────────────────────────
class Augmenter:
    """
    All augmentations operate on numpy arrays (crop_size, crop_size, crop_size).
    Applied only during training.
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_rotate: float = 0.5,
        p_zoom: float = 0.3,
        p_noise: float = 0.3,
        p_brightness: float = 0.3,
        p_blur: float = 0.2,
        p_gamma: float = 0.2,
        zoom_range: float = 0.2,
    ):
        self.p_flip       = p_flip
        self.p_rotate     = p_rotate
        self.p_zoom       = p_zoom
        self.p_noise      = p_noise
        self.p_brightness = p_brightness
        self.p_blur       = p_blur
        self.p_gamma      = p_gamma
        self.zoom_range   = zoom_range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Random flips along each axis
        for ax in range(3):
            if random.random() < self.p_flip:
                x = np.flip(x, axis=ax).copy()

        # Random 90° rotation in xy-plane
        if random.random() < self.p_rotate:
            k = random.choice([1, 2, 3])
            x = np.rot90(x, k=k, axes=(0, 1)).copy()

        # Random zoom (scale then re-crop to original size)
        if random.random() < self.p_zoom:
            x = self._random_zoom(x)

        # Gaussian noise
        if random.random() < self.p_noise:
            std = random.uniform(0.01, 0.08)
            x = x + np.random.normal(0, std, x.shape).astype(np.float32)

        # Brightness / contrast
        if random.random() < self.p_brightness:
            alpha = random.uniform(0.8, 1.2)   # contrast
            beta  = random.uniform(-0.1, 0.1)  # brightness
            x = alpha * x + beta

        # Gaussian blur
        if random.random() < self.p_blur:
            from scipy.ndimage import gaussian_filter
            sigma = random.uniform(0.5, 1.5)
            x = gaussian_filter(x, sigma=sigma).astype(np.float32)

        # Gamma correction
        if random.random() < self.p_gamma:
            x = self._gamma_correction(x)

        return x

    def _random_zoom(self, x: np.ndarray) -> np.ndarray:
        """Zoom in/out and re-crop to original size."""
        from scipy.ndimage import zoom as ndi_zoom
        factor = 1.0 + random.uniform(-self.zoom_range, self.zoom_range)
        zoomed = ndi_zoom(x, factor, order=1)
        return self._centre_crop_to(zoomed, x.shape[0])

    @staticmethod
    def _centre_crop_to(arr: np.ndarray, target: int) -> np.ndarray:
        """Centre-crop or pad a 3-D array to `target` on each axis."""
        result = np.zeros((target, target, target), dtype=arr.dtype)
        for ax in range(3):
            if arr.shape[ax] >= target:
                start = (arr.shape[ax] - target) // 2
                idx   = [slice(None)] * 3
                idx[ax] = slice(start, start + target)
                arr = arr[tuple(idx)]
        # After all axes, copy into output
        s = [min(arr.shape[i], target) for i in range(3)]
        result[:s[0], :s[1], :s[2]] = arr[:s[0], :s[1], :s[2]]
        return result

    @staticmethod
    def _gamma_correction(x: np.ndarray) -> np.ndarray:
        """Apply random gamma correction (shift to [0,1], apply, shift back)."""
        gamma = random.uniform(0.7, 1.5)
        mn, mx = x.min(), x.max()
        rng = mx - mn
        if rng < 1e-6:
            return x
        x_norm = (x - mn) / rng
        x_gamma = np.power(np.clip(x_norm, 1e-6, 1.0), gamma)
        return (x_gamma * rng + mn).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────
class EndometrialDataset(Dataset):
    """
    Loads CT crops for endometrial cancer grading.

    Each item returns:
        tensor of shape (1, crop_size, crop_size, crop_size)  — single channel
        int label  0=G1 / 1=G2 / 2=G3
    """

    def __init__(
        self,
        df: pd.DataFrame,
        crop_size: int = 96,
        crop_margin: int = 20,
        augment: bool = False,
    ):
        # Drop rows with missing image paths
        self.df = df.dropna(subset=["nifti_path"]).reset_index(drop=True)
        self.df = self.df[self.df["nifti_path"].astype(str).str.strip() != ""].reset_index(drop=True)
        self.crop_size   = crop_size
        self.crop_margin = crop_margin
        self.augment     = augment
        self.augmenter   = Augmenter() if augment else None

        log.info(
            f"  Dataset: {len(self.df)} cases | augment={augment} | "
            f"grade dist: { {g: int((self.df['grade']==g).sum()) for g in CLASS_NAMES} }"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = int(row["grade_int"])

        # ── Load volume ───────────────────────────────────────────────────
        try:
            volume = load_nifti(row["nifti_path"])
        except Exception as e:
            log.warning(f"Failed to load {row['nifti_path']}: {e}. Returning zeros.")
            volume = np.zeros(
                (self.crop_size, self.crop_size, self.crop_size), dtype=np.float32
            )
            return torch.from_numpy(volume[np.newaxis]).float(), label

        # ── Load mask (optional) ──────────────────────────────────────────
        mask_path = str(row.get("mask_path", "")).strip()
        has_mask  = mask_path not in ("", "nan") and Path(mask_path).exists()

        if has_mask:
            try:
                mask = load_nifti(mask_path)
            except Exception as e:
                log.warning(f"Failed to load mask {mask_path}: {e}. Using centre crop.")
                has_mask = False

        # ── Crop ──────────────────────────────────────────────────────────
        if has_mask:
            crop = crop_around_mask(volume, mask, self.crop_size, self.crop_margin)
        else:
            crop = centre_crop(volume, self.crop_size)

        # ── Normalise ─────────────────────────────────────────────────────
        crop = zscore_normalise(crop)

        # ── Augment ───────────────────────────────────────────────────────
        if self.augment and self.augmenter is not None:
            crop = self.augmenter(crop)

        # Add channel dim: (1, D, H, W)
        tensor = torch.from_numpy(crop[np.newaxis]).float()
        return tensor, label


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 3, pretrained: bool = False) -> nn.Module:
    """
    3D DenseNet-121 from MONAI.
    spatial_dims=3 → accepts (B, 1, D, H, W) volumes.
    out_channels=3 → outputs raw logits for G1/G2/G3.
    """
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze everything except the final classification layer."""
    for name, param in model.named_parameters():
        if "class_layers" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency class weights for cross-entropy loss.
    Handles G2-dominant imbalance in endometrial cancer datasets.
    """
    counts = df["grade_int"].value_counts().sort_index()
    total  = len(df)
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
        device=device,
    )
    log.info(f"  Class weights: { {CLASS_NAMES[i]: f'{weights[i].item():.3f}' for i in range(NUM_CLASSES)} }")
    return weights


def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    WeightedRandomSampler so each batch sees roughly balanced G1/G2/G3.
    """
    counts  = df["grade_int"].value_counts().sort_index()
    freq    = np.array([counts.get(i, 1) for i in range(NUM_CLASSES)])
    w_class = 1.0 / freq
    w_sample = np.array([w_class[int(g)] for g in df["grade_int"]])
    w_sample = w_sample / w_sample.sum()
    return WeightedRandomSampler(
        weights=torch.from_numpy(w_sample).float(),
        num_samples=len(df),
        replacement=True,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    training: bool,
) -> tuple[float, float, list, list]:
    """Run one epoch. Returns (avg_loss, accuracy, all_labels, all_probs)."""
    model.train() if training else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0
    all_labels = []
    all_probs  = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for inputs, labels in tqdm(loader, desc="  batch", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            logits = model(inputs)
            loss   = criterion(logits, labels)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += inputs.size(0)

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, all_labels, all_probs


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_accuracy(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history["train_loss"], label="Train loss", color="#2196F3")
    ax.plot(history["val_loss"],   label="Val loss",   color="#F44336")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss curves"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["train_acc"], label="Train acc", color="#2196F3")
    ax.plot(history["val_acc"],   label="Val acc",   color="#F44336")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy curves"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {save_path}")


def plot_confusion_matrix(labels: list, preds: list, save_path: Path):
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    # Annotate with count + %
    for i in range(3):
        for j in range(3):
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i,j]}\n({cm_pct[i,j]:.0f}%)",
                ha="center", va="center",
                fontsize=11,
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
            )
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
    ax.set_title("Confusion Matrix — Test Set (G1 / G2 / G3)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {save_path}")


def plot_roc_curves(labels: list, probs: list, save_path: Path):
    y_true_bin = label_binarize(labels, classes=[0, 1, 2])
    y_prob     = np.array(probs)

    colours = ["#4CAF50", "#2196F3", "#F44336"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, col) in enumerate(zip(CLASS_NAMES, colours)):
        try:
            auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls}  AUC = {auc:.3f}")
        except Exception:
            log.warning(f"  Could not compute ROC for {cls} (too few positives)")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One vs Rest (G1 / G2 / G3)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {save_path}")


def plot_per_class_metrics(labels: list, preds: list, save_path: Path):
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#FF9800", "#9C27B0", "#009688"]
    for k, (metric, col) in enumerate(zip(metrics, colours)):
        vals = [report[cls][metric] for cls in CLASS_NAMES]
        ax.bar(x + k * width, vals, width, label=metric.capitalize(), color=col, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score"); ax.set_title("Per-Class Metrics — Test Set")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    setup_logging(output_dir / "training.log")

    log.info("=" * 65)
    log.info("  Endometrial Cancer Grader — G1 / G2 / G3")
    log.info("=" * 65)
    log.info(f"  Labels CSV : {args.labels_csv}")
    log.info(f"  Output dir : {output_dir}")
    log.info(f"  Crop size  : {args.crop_size}³")
    log.info(f"  Crop margin: {args.crop_margin}")
    log.info(f"  Batch size : {args.batch_size}")
    log.info(f"  Epochs     : {args.epochs}")
    log.info(f"  LR         : {args.lr}")
    log.info(f"  Patience   : {args.patience}")
    log.info(f"  Seed       : {args.seed}")
    log.info(f"  Series     : {args.series if args.series else 'ALL'}") 

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device     : {device}")
    if device.type == "cuda":
        log.info(f"  GPU        : {torch.cuda.get_device_name(0)}")

    # ── Load labels ──────────────────────────────────────────────────────────
    df = pd.read_csv(args.labels_csv)
    log.info(f"\nLoaded {len(df)} rows | {df['patient_id'].nunique()} patients")

    # Keep only rows with valid grade
    df = df.dropna(subset=["grade_int"]).copy()
    df["grade_int"] = df["grade_int"].astype(int)

    # Validate grade values
    invalid = df[~df["grade_int"].isin([0, 1, 2])]
    if len(invalid) > 0:
        log.warning(f"Dropping {len(invalid)} rows with invalid grade_int: {invalid['grade_int'].unique()}")
        df = df[df["grade_int"].isin([0, 1, 2])].copy()

    # ── Optional: filter by series category ──────────────────────────────────
    if args.series:
        requested = [s.strip() for s in args.series.split(",")]
        df = df[df["series_category"].isin(requested)].reset_index(drop=True)
        log.info(f"  Series filter: {requested} -> {len(df)} rows remain")
    else:
        log.info(f"  No series filter — using all {len(df)} scan rows")

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    # If no explicit val split exists (e.g. CPTAC-only CSV has only train/test),
    # carve 15% out of train at patient level, stratified by grade.
    # This ensures early stopping and best-model logic always have a val set.
    if len(val_df) == 0:
        log.warning("No 'val' split found in CSV — auto-carving 15% from train "
                    "(patient-level, stratified by grade).")
        from sklearn.model_selection import train_test_split as _tts
        train_patients = (train_df[["patient_id", "grade"]]
                          .drop_duplicates("patient_id")
                          .reset_index(drop=True))
        try:
            tr_pids, va_pids = _tts(
                train_patients["patient_id"],
                test_size=0.15,
                stratify=train_patients["grade"],
                random_state=42,
            )
        except ValueError:
            # Fallback: random split if stratify fails (e.g. tiny dataset)
            tr_pids, va_pids = _tts(
                train_patients["patient_id"],
                test_size=0.15,
                random_state=42,
            )
        val_df   = train_df[train_df["patient_id"].isin(va_pids)].reset_index(drop=True)
        train_df = train_df[train_df["patient_id"].isin(tr_pids)].reset_index(drop=True)
        log.info(f"  Auto val: {len(val_df)} scans carved from train; "
                 f"remaining train: {len(train_df)} scans")

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pts = split_df["patient_id"].nunique()
        log.info(f"\n{split_name}: {len(split_df)} scans | {pts} patients")
        for g in CLASS_NAMES:
            n_pts  = split_df[split_df["grade"] == g]["patient_id"].nunique()
            n_rows = (split_df["grade"] == g).sum()
            log.info(f"  {g}: {n_pts} patients / {n_rows} scans")

    if len(train_df) == 0:
        log.error("No training cases found. Check labels CSV and 'split' column.")
        return

    # ── Datasets & loaders ───────────────────────────────────────────────────
    log.info("\nBuilding datasets...")
    train_dataset = EndometrialDataset(
        train_df, args.crop_size, args.crop_margin, augment=True
    )
    val_dataset = EndometrialDataset(
        val_df, args.crop_size, args.crop_margin, augment=False
    )
    test_dataset = EndometrialDataset(
        test_df, args.crop_size, args.crop_margin, augment=False
    )

    sampler = make_sampler(train_df)
    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("\nBuilding 3D DenseNet-121 (3-class)...")
    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total params    : {total_params:,}")
    log.info(f"  Trainable params: {trainable_params:,}")

    # Weighted cross-entropy loss
    class_weights = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # ── Training history ──────────────────────────────────────────────────────
    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    best_val_loss = float("inf")
    best_bal_acc  = 0.0
    patience_counter = 0

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 1: warm up classification head only (5 epochs)
    # ══════════════════════════════════════════════════════════════════════════
    WARMUP_EPOCHS = 5
    log.info(f"\n{'='*65}")
    log.info(f"  PHASE 1 — Warm-up head ({WARMUP_EPOCHS} epochs, backbone frozen)")
    log.info(f"{'='*65}")

    freeze_backbone(model)
    optimizer_p1 = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 10,   # higher LR for head-only
    )

    for epoch in range(1, WARMUP_EPOCHS + 1):
        log.info(f"\nWarmup epoch {epoch}/{WARMUP_EPOCHS}")
        t_loss, t_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer_p1, device, training=True
        )
        v_loss, v_acc, v_labels, v_probs = run_epoch(
            model, val_loader, criterion, optimizer_p1, device, training=False
        )
        v_preds = [int(np.argmax(p)) for p in v_probs]
        bal_acc = balanced_accuracy_score(v_labels, v_preds) if v_labels else 0.0

        log.info(f"  train loss={t_loss:.4f}  acc={t_acc:.3f}")
        log.info(f"  val   loss={v_loss:.4f}  acc={v_acc:.3f}  bal_acc={bal_acc:.3f}")

        writer.add_scalars("Loss",         {"train": t_loss, "val": v_loss}, epoch)
        writer.add_scalars("Accuracy",     {"train": t_acc,  "val": v_acc},  epoch)
        writer.add_scalar("BalancedAcc/val", bal_acc, epoch)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase 2: fine-tune full network
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*65}")
    log.info(f"  PHASE 2 — Fine-tune full network ({args.epochs} epochs)")
    log.info(f"{'='*65}")

    unfreeze_all(model)
    optimizer_p2 = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler    = CosineAnnealingLR(optimizer_p2, T_max=args.epochs, eta_min=1e-6)

    for epoch in range(1, args.epochs + 1):
        global_epoch = WARMUP_EPOCHS + epoch
        log.info(f"\nEpoch {epoch}/{args.epochs}  (global {global_epoch})")

        t_loss, t_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer_p2, device, training=True
        )
        v_loss, v_acc, v_labels, v_probs = run_epoch(
            model, val_loader, criterion, optimizer_p2, device, training=False
        )
        scheduler.step()

        v_preds   = [int(np.argmax(p)) for p in v_probs]
        bal_acc   = balanced_accuracy_score(v_labels, v_preds) if v_labels else 0.0
        current_lr = scheduler.get_last_lr()[0]

        log.info(f"  train loss={t_loss:.4f}  acc={t_acc:.3f}")
        log.info(f"  val   loss={v_loss:.4f}  acc={v_acc:.3f}  bal_acc={bal_acc:.3f}  lr={current_lr:.2e}")

        writer.add_scalars("Loss",           {"train": t_loss, "val": v_loss},  global_epoch)
        writer.add_scalars("Accuracy",       {"train": t_acc,  "val": v_acc},   global_epoch)
        writer.add_scalar("BalancedAcc/val", bal_acc, global_epoch)
        writer.add_scalar("LR",             current_lr, global_epoch)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        # ── Save best model (by val loss) ─────────────────────────────────
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_bal_acc  = bal_acc
            patience_counter = 0
            ckpt_path = output_dir / "best_classifier.pth"
            torch.save(
                {
                    "epoch":      global_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer_p2.state_dict(),
                    "val_loss":   v_loss,
                    "val_acc":    v_acc,
                    "bal_acc":    bal_acc,
                    "args":       vars(args),
                },
                ckpt_path,
            )
            log.info(f"  ✓ Saved best model  (val_loss={v_loss:.4f}  bal_acc={bal_acc:.3f})")
        else:
            patience_counter += 1
            log.info(f"  No improvement. Patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                log.info(f"  Early stopping at epoch {epoch}.")
                break

    writer.close()

    # ── Save training history ──────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "training_history.csv", index=False)
    log.info(f"\nTraining history saved -> {output_dir / 'training_history.csv'}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Final evaluation on test set using best model
    # ─────────────────────────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  FINAL EVALUATION — loading best checkpoint")
    log.info(f"{'='*65}")

    ckpt = torch.load(output_dir / "best_classifier.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"  Best checkpoint from epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

    _, _, final_labels, final_probs = run_epoch(
        model, test_loader, criterion, None, device, training=False
    )
    final_preds = [int(np.argmax(p)) for p in final_probs]

    bal_acc = balanced_accuracy_score(final_labels, final_preds)
    report  = classification_report(
        final_labels, final_preds,
        target_names=CLASS_NAMES, zero_division=0
    )

    log.info(f"\n  Balanced accuracy : {bal_acc:.4f}")
    log.info(f"\n{report}")

    # Per-class AUC
    if len(set(final_labels)) > 1:
        y_true_bin = label_binarize(final_labels, classes=[0, 1, 2])
        y_prob_arr = np.array(final_probs)
        for i, cls in enumerate(CLASS_NAMES):
            try:
                auc = roc_auc_score(y_true_bin[:, i], y_prob_arr[:, i])
                log.info(f"  {cls} AUC: {auc:.4f}")
            except Exception:
                log.warning(f"  {cls} AUC: could not compute (too few positives)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    log.info("\nGenerating evaluation plots...")
    plot_loss_accuracy(history, plot_dir / "loss_accuracy_curves.png")
    plot_confusion_matrix(final_labels, final_preds, plot_dir / "confusion_matrix.png")
    plot_roc_curves(final_labels, final_probs, plot_dir / "roc_curves.png")
    plot_per_class_metrics(final_labels, final_preds, plot_dir / "per_class_metrics.png")

    # ── Save config ───────────────────────────────────────────────────────────
    config = vars(args)
    config.update(
        {
            "best_val_loss":  float(best_val_loss),
            "best_bal_acc":   float(best_bal_acc),
            "final_bal_acc":  float(bal_acc),
            "n_train":        len(train_df),
            "n_val":          len(val_df),
            "n_test":         len(test_df),
            "series_filter":  args.series if args.series else "ALL",
            "device":         str(device),
            "class_names":    CLASS_NAMES,
        }
    )
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Config saved -> {output_dir / 'config.json'}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best val loss     : {best_val_loss:.4f}")
    print(f"  Balanced accuracy : {bal_acc:.4f}  ({bal_acc*100:.1f}%)")
    print(f"  Random chance     : 33.3%")
    print()
    print(f"  Outputs saved to  : {output_dir}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train 3D DenseNet-121 for G1/G2/G3 endometrial cancer grading"
    )
    p.add_argument(
        "--labels_csv", required=True,
        help="Path to labels_simple.csv or combined_labels_simple.csv"
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory to save model, plots, and logs"
    )
    p.add_argument(
        "--crop_size", type=int, default=96,
        help="Cube side length for tumour crop in voxels (default: 96)"
    )
    p.add_argument(
        "--crop_margin", type=int, default=20,
        help="Extra voxels around bounding box when cropping (default: 20)"
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="Training batch size (default: 4). Reduce to 2 if OOM."
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="Max fine-tuning epochs in Phase 2 (default: 50)"
    )
    p.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for Phase 2 fine-tuning (default: 1e-4)"
    )
    p.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience in epochs (default: 15)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    p.add_argument(
        "--series", type=str, default=None,
        help=(
            "Comma-separated series categories to train on. "
            "e.g. 'AP_Routine' or 'AP_Routine,Venous,Pre_Contrast'. "
            "Default: use ALL series in the CSV."
        )
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())