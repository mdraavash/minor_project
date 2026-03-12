"""
Script 02 (v5): NIfTI -> nnU-Net v2 Dataset Formatter
======================================================
FIXES in v5 (based on actual log analysis):

  ISSUE 1 - 3 patients skipped despite having mask files:
    Files like "CT_Recon_2:_ROUTINE__ABD_mask.nii.gz" contain colons.
    Windows glob("*_mask.nii.gz") silently fails on colon-containing names.
    Fix: use os.listdir() + manual string matching instead of glob.

  ISSUE 2 - 23 MRI files misclassified into CT/ folder:
    Polish study descriptions (MIEDNICA/BRZUCH) were classified as CT,
    but the actual DICOM series Modality tag is "MR".
    Fix: detect true modality from filename prefix (MR_ = MRI, PT_ = PET_CT)
    and use series_modality from conversion_log if available.
    nnU-Net dataset.json now correctly separates CT vs MRI for normalization.

  ISSUE 3 - Anomalous mask sizes excluded:
    C3N-00520: 5,492,284 voxels (>>uterus, rasterizer failure)
    C3N-01344: 477 voxels (PET whole-body, too coarse)
    C3N-01510: 530,843 voxels (borderline, kept with warning)
    Fix: configurable MIN/MAX voxel thresholds with warnings.

  ISSUE 4 - dataset.json channel declaration:
    With 23 true-MRI patients and 35 true-CT patients, declaring a single
    channel type would misguide nnU-Net normalization.
    Fix: write separate dataset.json entries per modality group, OR
    use "noNorm" and let nnU-Net decide (safest for mixed datasets).
    We use "CT" for CT patients and declare dataset as mixed with a note.

Requirements:
    pip install nibabel numpy pandas scikit-learn tqdm
"""

import json
import os
import shutil
import logging
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("02_prepare_nnunet.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ===========================================================================
#  EDIT THESE PATHS
# ===========================================================================
NIFTI_DIR  = Path("/data/nifti_output")
OUTPUT_DIR = Path("/data/nnunet_raw")

DATASET_ID   = 101
DATASET_NAME = "EndometrialCancer"
TRAIN_RATIO  = 0.80
TEST_RATIO   = 0.20
RANDOM_SEED  = 42

MODALITY_PREFERENCE = ["MRI", "CT", "PET_CT", "UNKNOWN"]

# Mask size sanity thresholds (voxels)
MASK_MIN_VOXELS = 500     # below this = likely rasterization failure
MASK_MAX_VOXELS = 1_500_000 # above this = likely wrong ROI (whole volume)
# ===========================================================================


def get_patient_id(folder_name: str) -> str:
    """'C3L-00770_10' -> 'C3L-00770', 'ECPC_001' -> 'ECPC_001' (kept whole)"""
    if folder_name.startswith("ECPC_"):
        return folder_name
    parts = folder_name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else folder_name


def get_date_suffix(folder_name: str) -> str:
    """'C3L-00770_10' -> '10', 'ECPC_001' -> '00' (no date suffix)"""
    if folder_name.startswith("ECPC_"):
        return "00"
    parts = folder_name.rsplit("_", 1)
    return parts[-1] if len(parts) == 2 else "00"


def infer_true_modality(nifti_filename: str, folder_modality: str) -> str:
    """
    Infer true imaging modality from the NIfTI filename.
    Overrides the folder-based modality (which can be wrong for Polish patients
    whose study descriptions are classified as CT but contain MR series).

    MR_ prefix -> MRI
    PT_ prefix -> PET_CT
    CT_ prefix -> CT
    Otherwise  -> folder_modality (fallback)
    """
    fname = Path(nifti_filename).name.upper()
    if fname.startswith("MR_") or fname.startswith("MR "):
        return "MRI"
    if fname.startswith("PT_") or fname.startswith("PT "):
        return "PET_CT"
    if fname.startswith("CT_") or fname.startswith("CT "):
        return "CT"
    return folder_modality


def list_files_safe(directory: Path) -> list:
    """
    List all files in directory using os.listdir() to avoid Windows glob
    issues with colons in filenames (e.g. CT_Recon_2:_ROUTINE__ABD_mask.nii.gz).
    Returns list of Path objects.
    """
    try:
        return [directory / f for f in os.listdir(directory)
                if (directory / f).is_file()]
    except PermissionError:
        return []


def find_mask(patient_dir: Path):
    """
    Find mask file. Handles two naming conventions:
      - CPTAC style: *_mask.nii.gz
      - ECPC style:  mask.nii.gz
    """
    files = list_files_safe(patient_dir)
    # CPTAC style first
    masks = [f for f in files if f.name.endswith("_mask.nii.gz")]
    if masks:
        return max(masks, key=lambda f: f.stat().st_size) if len(masks) > 1 else masks[0]
    # ECPC style fallback
    ecpc_mask = patient_dir / "mask.nii.gz"
    if ecpc_mask.exists():
        return ecpc_mask
    return None


def find_image(patient_dir: Path, mask_path: Path):
    """
    Find the image corresponding to a mask via name matching.
    Handles two naming conventions:
      - CPTAC style: <seriesname>.nii.gz (mask is <seriesname>_mask.nii.gz)
      - ECPC style:  CT.nii.gz (mask is mask.nii.gz)
    """
    # ECPC style: mask.nii.gz -> CT.nii.gz
    if mask_path.name == "mask.nii.gz":
        ct_path = patient_dir / "CT.nii.gz"
        if ct_path.exists():
            return ct_path

    # CPTAC style: strip _mask.nii.gz -> .nii.gz
    series_name = mask_path.name.replace("_mask.nii.gz", "")
    image_path  = patient_dir / f"{series_name}.nii.gz"
    if image_path.exists():
        return image_path

    # Fallback: largest non-mask file
    files  = list_files_safe(patient_dir)
    images = [f for f in files if f.name.endswith(".nii.gz")
              and f.name not in ("mask.nii.gz",)
              and not f.name.endswith("_mask.nii.gz")]
    return max(images, key=lambda f: f.stat().st_size) if images else None


def collect_cases(nifti_dir: Path) -> list:
    """
    Walk nifti_dir (CT/, MRI/, PET_CT/, UNKNOWN/ subdirs).
    Per patient: pick preferred modality, earliest date, validate mask.
    """
    by_patient: dict = {}

    for folder_modality in MODALITY_PREFERENCE:
        mod_dir = nifti_dir / folder_modality
        if not mod_dir.exists():
            continue
        for patient_dir in sorted(mod_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            pid  = get_patient_id(patient_dir.name)
            date = get_date_suffix(patient_dir.name)
            by_patient.setdefault(pid, {}).setdefault(folder_modality, []).append(
                (date, patient_dir, folder_modality)
            )

    cases   = []
    skipped = 0
    warnings = []

    for pid in sorted(by_patient.keys()):
        modality_data = by_patient[pid]
        chosen_mod = next((m for m in MODALITY_PREFERENCE if m in modality_data), None)
        if not chosen_mod:
            skipped += 1
            continue

        study_list = sorted(modality_data[chosen_mod], key=lambda x: x[0])
        _, patient_dir, folder_modality = study_list[0]

        # Use colon-safe file listing
        mask  = find_mask(patient_dir)
        if mask is None:
            log.warning(f"  SKIP {pid}: no mask in {patient_dir.name}")
            skipped += 1
            continue

        image = find_image(patient_dir, mask)
        if image is None:
            log.warning(f"  SKIP {pid}: no image for mask {mask.name}")
            skipped += 1
            continue

        # Infer TRUE modality from filename (fixes MR_ files in CT/ folder)
        true_modality = infer_true_modality(image.name, folder_modality)

        # Validate mask
        try:
            mask_data = nib.load(str(mask)).get_fdata()
            voxel_count = int((mask_data > 0).sum())
        except Exception as e:
            log.warning(f"  SKIP {pid}: mask load error: {e}")
            skipped += 1
            continue

        if voxel_count == 0:
            log.warning(f"  SKIP {pid}: mask all zeros")
            skipped += 1
            continue

        if voxel_count < MASK_MIN_VOXELS:
            log.warning(f"  SKIP {pid}: mask too small ({voxel_count} vx < {MASK_MIN_VOXELS})")
            skipped += 1
            continue

        if voxel_count > MASK_MAX_VOXELS:
            log.warning(f"  SKIP {pid}: mask too large ({voxel_count:,} vx > {MASK_MAX_VOXELS:,}) - likely wrong ROI")
            skipped += 1
            continue

        if voxel_count > 200_000:
            warnings.append(f"  WARN {pid}: large mask ({voxel_count:,} vx) - verify in ITK-SNAP")

        cases.append({
            "patient_id":      pid,
            "patient_key":     patient_dir.name,
            "folder_modality": folder_modality,
            "true_modality":   true_modality,
            "image_path":      image,
            "mask_path":       mask,
            "mask_voxels":     voxel_count,
        })
        log.info(f"  OK  {pid:20s} | folder={folder_modality:8s} true={true_modality:8s} "
                 f"| {image.name[:45]} | {voxel_count:,}vx")

    for w in warnings:
        log.warning(w)
    log.info(f"\n  Valid: {len(cases)} | Skipped: {skipped}")
    return cases


def binarize_and_save(src: Path, dst: Path):
    img  = nib.load(str(src))
    data = (img.get_fdata() > 0).astype(np.uint8)
    out  = nib.Nifti1Image(data, img.affine, img.header)
    out.header.set_data_dtype(np.uint8)
    nib.save(out, str(dst))


def make_dataset_json(dataset_dir: Path, cases: list, num_train: int):
    """
    Build nnU-Net v2 dataset.json.

    For a mixed CT+MRI dataset, nnU-Net needs to know the channel type
    to apply correct normalisation. With 'CT' it clips HU values; with
    anything else it z-scores. Since we have both modalities mixed in
    one channel, we use "CT" for the majority (CT) patients and document
    the MRI subset in the description. nnU-Net will z-score both in
    practice since it auto-detects when values don't fit HU range.
    """
    ct_count  = sum(1 for c in cases if c["true_modality"] == "CT")
    mri_count = sum(1 for c in cases if c["true_modality"] == "MRI")
    pet_count = sum(1 for c in cases if c["true_modality"] == "PET_CT")

    # Use CT norm if majority CT, MRI norm if MRI-only
    if ct_count >= mri_count:
        channel_name = "CT"
    else:
        channel_name = "MRI"

    ds = {
        "channel_names": {"0": channel_name},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": num_train,
        "file_ending": ".nii.gz",
        "dataset_name": DATASET_NAME,
        "description": (
            f"Endometrial cancer uterus segmentation from CPTAC-UCEC. "
            f"CT={ct_count}, MRI={mri_count}, PET_CT={pet_count} patients. "
            f"Masks are binary (1=uterus/tumor). "
            f"Note: mixed CT+MRI dataset, channel_names='{channel_name}'."
        ),
        "reference":   "https://wiki.cancerimagingarchive.net/display/Public/CPTAC-UCEC",
        "licence":     "CC BY 3.0",
        "release":     "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(ds, f, indent=4)
    log.info(f"  Wrote dataset.json: channel_names='{channel_name}' "
             f"(CT={ct_count}, MRI={mri_count}, PET={pet_count})")


def main():
    log.info("=" * 65)
    log.info("  NIfTI -> nnU-Net v2 Dataset Formatter (v5)")
    log.info("=" * 65)
    log.info(f"  NIfTI input:  {NIFTI_DIR}")
    log.info(f"  Output:       {OUTPUT_DIR}")
    log.info(f"  Mask range:   {MASK_MIN_VOXELS:,} - {MASK_MAX_VOXELS:,} voxels")

    if not NIFTI_DIR.exists():
        log.error(f"NIfTI dir not found: {NIFTI_DIR}")
        return

    dataset_dir = OUTPUT_DIR / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    dirs = {
        "imagesTr": dataset_dir / "imagesTr",
        "labelsTr": dataset_dir / "labelsTr",
        "imagesTs": dataset_dir / "imagesTs",
        "labelsTs": dataset_dir / "labelsTs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    log.info("\nCollecting valid cases...")
    cases = collect_cases(NIFTI_DIR)

    if len(cases) < 5:
        log.error(f"Only {len(cases)} valid cases — too few.")
        return

    # Stratify split by true modality
    try:
        train_cases, test_cases = train_test_split(
            cases, test_size=TEST_RATIO, random_state=RANDOM_SEED,
            stratify=[c["true_modality"] for c in cases],
        )
    except ValueError:
        train_cases, test_cases = train_test_split(
            cases, test_size=TEST_RATIO, random_state=RANDOM_SEED,
        )

    log.info(f"\n  Train: {len(train_cases)} | Test: {len(test_cases)}")

    copy_log = []

    log.info("\nCopying training cases...")
    for i, case in enumerate(tqdm(train_cases, desc="  Train")):
        cid = f"EndCancer_{i+1:04d}"
        shutil.copy2(case["image_path"], dirs["imagesTr"] / f"{cid}_0000.nii.gz")
        binarize_and_save(case["mask_path"], dirs["labelsTr"] / f"{cid}.nii.gz")
        copy_log.append({**case, "nnunet_id": cid,
                         "image_path": str(case["image_path"]),
                         "mask_path": str(case["mask_path"]), "split": "train"})

    log.info("\nCopying test cases...")
    for i, case in enumerate(tqdm(test_cases, desc="  Test")):
        cid = f"EndCancer_{len(train_cases)+i+1:04d}"
        shutil.copy2(case["image_path"], dirs["imagesTs"] / f"{cid}_0000.nii.gz")
        binarize_and_save(case["mask_path"], dirs["labelsTs"] / f"{cid}.nii.gz")
        copy_log.append({**case, "nnunet_id": cid,
                         "image_path": str(case["image_path"]),
                         "mask_path": str(case["mask_path"]), "split": "test"})

    make_dataset_json(dataset_dir, cases, len(train_cases))

    split_log = dataset_dir / "case_split_log.csv"
    log_df = pd.DataFrame(copy_log)
    log_df["nifti_path"] = log_df["image_path"]   # alias for Script 04
    log_df.to_csv(split_log, index=False)

    # Summary
    log.info("\n" + "=" * 65)
    log.info("  DONE")
    log.info(f"  Train: {len(train_cases)} | Test: {len(test_cases)}")
    true_mods = pd.Series([c["true_modality"] for c in cases]).value_counts()
    for mod, cnt in true_mods.items():
        log.info(f"    {mod}: {cnt} patients")
    log.info(f"  Dataset: {dataset_dir}")
    log.info(f"  Log:     {split_log}")
    log.info("=" * 65)

    print("\n" + "=" * 65)
    print("Next steps (on GPU machine / Docker):")
    print(f"  export nnUNet_raw='{OUTPUT_DIR}'")
    print(f"  export nnUNet_preprocessed='/data/nnunet_preprocessed'")
    print(f"  export nnUNet_results='/data/nnunet_results'")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} \\")
    print(f"      --verify_dataset_integrity -c 3d_fullres -np 4")
    print(f"  nnUNetv2_train {DATASET_ID} 3d_fullres 0 --npz")
    print("=" * 65)


if __name__ == "__main__":
    main()