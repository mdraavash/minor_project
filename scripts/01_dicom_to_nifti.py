"""
Script 01 (v5): DICOM + RTStruct -> NIfTI Converter
====================================================
FIXED (v5):
  - Replaced dcmrtstruct2nii entirely with a pure pydicom RTStruct rasterizer.
    dcmrtstruct2nii was producing masks covering the entire abdomen because
    it failed to correctly map contour coordinates to voxel space.
  - The new rasterizer:
      1. Reads RTStruct contour points (DICOM patient coords, mm)
      2. Reads each imaging DICOM slice for its ImagePositionPatient + 
         PixelSpacing + ImageOrientationPatient
      3. Matches each contour to the correct slice by z-position
      4. Rasterizes each polygon with skimage.draw.polygon2 in voxel space
      5. Saves a clean binary NIfTI aligned to the reference image
  - Groups all ROI entries per patient+study, picks UTERUS first, writes 
    exactly ONE mask per image series (no overwriting by lymph node ROIs)

Requirements:
    pip install pydicom SimpleITK nibabel tqdm pandas numpy scikit-image
    (dcmrtstruct2nii no longer required)
"""

import sys
import shutil
import logging
from pathlib import Path

import pydicom
import SimpleITK as sitk
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("conversion_run.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ===========================================================================
#  EDIT THESE PATHS
# ===========================================================================
DATASET_ROOT  = Path(r"F:\dataset")
OUTPUT_DIR    = Path(r"F:\nifti_output")
PRE_DOSE_ONLY = True
RESUME        = True
TUMOR_ROI_KEYWORDS = [
    "uterus", "tumor", "gtv", "ctv", "primary", "mass", "lesion", "cervix"
]
# ===========================================================================


# ---------------------------------------------------------------------------
#  Pure-pydicom RTStruct rasterizer
# ---------------------------------------------------------------------------

def _get_roi_name_map(rtstruct_ds) -> dict:
    """Return {ROI Number -> ROI Name} from StructureSetROISequence."""
    return {
        roi.ROINumber: roi.ROIName
        for roi in getattr(rtstruct_ds, "StructureSetROISequence", [])
    }


def list_rt_structs_pydicom(rtstruct_file: Path) -> list:
    """List all ROI names in an RTStruct DICOM file."""
    try:
        ds = pydicom.dcmread(str(rtstruct_file), stop_before_pixels=True)
        return [roi.ROIName for roi in getattr(ds, "StructureSetROISequence", [])]
    except Exception as e:
        log.warning(f"    Could not read RTStruct names: {e}")
        return []


def _build_slice_map(dicom_files: list) -> dict:
    """
    Read each imaging DICOM slice and build a map:
        { z_position_mm (float) -> slice metadata dict }
    Slice metadata: ipp, pixel_spacing, orientation, rows, cols, instance_number
    """
    slices = {}
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if not hasattr(ds, "ImagePositionPatient"):
                continue
            ipp = [float(v) for v in ds.ImagePositionPatient]
            z   = ipp[2]
            ps  = [float(v) for v in ds.PixelSpacing] if hasattr(ds, "PixelSpacing") else [1.0, 1.0]
            iop = ([float(v) for v in ds.ImageOrientationPatient]
                   if hasattr(ds, "ImageOrientationPatient")
                   else [1,0,0,0,1,0])
            slices[z] = {
                "ipp": ipp,
                "pixel_spacing": ps,   # [row_spacing, col_spacing]
                "orientation": iop,    # 6 direction cosines
                "rows": int(ds.Rows),
                "cols": int(ds.Columns),
                "instance": int(getattr(ds, "InstanceNumber", 0)),
            }
        except Exception:
            pass
    return slices


def _mm_to_pixel(x_mm, y_mm, ipp, iop, ps):
    """
    Convert a single (x,y,z) patient-coordinate point to (row, col) pixel coords.
    ipp: ImagePositionPatient [x,y,z]
    iop: ImageOrientationPatient [F1x,F1y,F1z, F2x,F2y,F2z]
    ps:  PixelSpacing [row_spacing, col_spacing]
    Returns (row, col) as floats.
    """
    F1 = np.array(iop[:3])  # row direction cosine
    F2 = np.array(iop[3:])  # col direction cosine
    delta = np.array([x_mm, y_mm, 0.0]) - np.array(ipp[:2] + [0.0])
    # Project onto row/col axes
    # Note: RTStruct x,y are in the same plane as iop
    # col = dot(point - ipp, F1) / col_spacing
    # row = dot(point - ipp, F2) / row_spacing
    pt = np.array([x_mm - ipp[0], y_mm - ipp[1], 0.0])
    col = np.dot(pt, F1) / ps[1]
    row = np.dot(pt, F2) / ps[0]
    return row, col


def rasterize_rtstruct(rtstruct_file: Path, dicom_files: list,
                        ref_nifti_path: Path, roi_name: str) -> np.ndarray | None:
    """
    Rasterize a single ROI from an RTStruct file into a binary numpy mask
    aligned with the reference NIfTI volume.

    Returns uint8 numpy array shape (X, Y, Z) matching the NIfTI, or None on failure.
    """
    try:
        from skimage.draw import polygon2mask
    except ImportError:
        # fallback to polygon
        from skimage.draw import polygon as sk_polygon
        polygon2mask = None

    # Load RTStruct
    try:
        rt_ds = pydicom.dcmread(str(rtstruct_file))
    except Exception as e:
        log.error(f"    Cannot read RTStruct: {e}")
        return None

    roi_name_map = _get_roi_name_map(rt_ds)

    # Find the ROIContourSequence entry matching our ROI name
    target_contour_seq = None
    for rcs in getattr(rt_ds, "ROIContourSequence", []):
        rn = roi_name_map.get(rcs.ReferencedROINumber, "")
        if rn.strip().upper() == roi_name.strip().upper():
            target_contour_seq = rcs
            break

    if target_contour_seq is None:
        log.warning(f"    ROI '{roi_name}' not found in ROIContourSequence")
        return None

    contours = getattr(target_contour_seq, "ContourSequence", [])
    if not contours:
        log.warning(f"    ROI '{roi_name}' has no ContourSequence entries")
        return None

    # Build slice geometry map from imaging DICOMs
    slice_map = _build_slice_map(dicom_files)
    if not slice_map:
        log.error("    Could not build slice map from imaging DICOMs")
        return None

    sorted_z = sorted(slice_map.keys())
    slice_spacing = abs(sorted_z[1] - sorted_z[0]) if len(sorted_z) > 1 else 1.0

    # Load reference NIfTI to get output shape
    ref_img   = nib.load(str(ref_nifti_path))
    ref_shape = ref_img.shape[:3]   # (X, Y, Z) in NIfTI convention
    n_slices  = ref_shape[2]

    # Map z -> slice index using the NIfTI affine
    # SimpleITK writes NIfTI so that slice index k corresponds to
    # the k-th element along the z-axis of sorted_z
    # (ascending or descending depending on scanner)
    z_to_k = {}
    for k, z in enumerate(sorted_z):
        z_to_k[z] = k

    # Initialize mask (X, Y, Z)
    mask = np.zeros(ref_shape, dtype=np.uint8)

    n_contours_placed = 0
    for contour in contours:
        geom_type = getattr(contour, "ContourGeometricType", "")
        data      = list(map(float, contour.ContourData))
        if len(data) < 9:
            continue   # need at least 3 points

        # RTStruct contour points: flat list [x1,y1,z1, x2,y2,z2, ...]
        pts = np.array(data).reshape(-1, 3)
        z_val = pts[0, 2]

        # Find nearest slice
        nearest_z = min(sorted_z, key=lambda z: abs(z - z_val))
        if abs(nearest_z - z_val) > slice_spacing * 1.5:
            log.debug(f"    Contour z={z_val:.2f} has no close slice (nearest {nearest_z:.2f})")
            continue

        k = z_to_k[nearest_z]
        if k < 0 or k >= n_slices:
            continue

        sl = slice_map[nearest_z]
        ipp = sl["ipp"]
        iop = sl["orientation"]
        ps  = sl["pixel_spacing"]
        rows = sl["rows"]
        cols = sl["cols"]

        # Convert all contour points to pixel coords
        pixel_pts = []
        for pt in pts:
            r, c = _mm_to_pixel(pt[0], pt[1], ipp, iop, ps)
            pixel_pts.append((r, c))

        pixel_pts = np.array(pixel_pts)
        rows_arr = np.clip(pixel_pts[:, 0], 0, rows - 1).astype(int)
        cols_arr = np.clip(pixel_pts[:, 1], 0, cols - 1).astype(int)

        # Rasterize polygon onto a 2D slice
        if polygon2mask is not None:
            # polygon2mask wants (row, col) pairs and shape
            try:
                poly_mask = polygon2mask(
                    (rows, cols),
                    np.column_stack([rows_arr, cols_arr])
                ).astype(np.uint8)
            except Exception:
                poly_mask = np.zeros((rows, cols), dtype=np.uint8)
        else:
            from skimage.draw import polygon as sk_polygon
            rr, cc = sk_polygon(rows_arr, cols_arr, shape=(rows, cols))
            poly_mask = np.zeros((rows, cols), dtype=np.uint8)
            poly_mask[rr, cc] = 1

        # Place into 3D mask
        # NIfTI from SimpleITK: shape is (X, Y, Z) where X=cols, Y=rows
        # poly_mask is (rows, cols) -> need to transpose to (cols, rows) = (X, Y)
        slice_2d = poly_mask.T  # now (X, Y) = (cols, rows)

        if slice_2d.shape == (ref_shape[0], ref_shape[1]):
            mask[:, :, k] = np.logical_or(mask[:, :, k], slice_2d).astype(np.uint8)
            n_contours_placed += 1
        else:
            log.debug(f"    Shape mismatch: slice {slice_2d.shape} vs ref {ref_shape[:2]}")

    log.info(f"    Rasterized {n_contours_placed}/{len(contours)} contours for '{roi_name}'")
    nonzero = int(mask.sum())
    log.info(f"    Nonzero voxels: {nonzero}")

    if nonzero == 0:
        log.warning(f"    Mask is empty after rasterization!")

    return mask


# ---------------------------------------------------------------------------
#  Rest of the pipeline (unchanged from v4)
# ---------------------------------------------------------------------------

def find_manifest_dirs(root: Path) -> list:
    manifests = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("manifest-"):
            csv = d / "metadata.csv"
            if csv.exists():
                manifests.append((d, csv))
                log.info(f"  Found manifest: {d.name} (metadata.csv OK)")
            else:
                log.warning(f"  Manifest folder {d.name} has no metadata.csv, skipping")
    return manifests


def parse_windows_path(raw_path: str, manifest_dir: Path) -> Path:
    normalized = raw_path.replace("\\\\", "/").replace("\\", "/").lstrip("./")
    return manifest_dir / Path(normalized)


def infer_modality(study_description: str) -> str:
    s = str(study_description).upper()
    if "PET" in s:
        return "PET_CT"
    if any(k in s for k in ["MR ", "MRI", "MAGNETIC", "T1", "T2",
                              "PELVIS WITH", "PELVIS WITHOUT"]):
        return "MRI"
    if any(k in s for k in ["CT", "ABDOMEN", "PELVIS", "THORAX", "CHEST",
                              "MIEDNICA", "BRZUCH", "KLATKA", "UROGRAPHY",
                              "CAP", "RENAL", "BODYMIEDNICA"]):
        return "CT"
    return "UNKNOWN"


def convert_series_to_nifti(series_files: list, output_path: Path) -> bool:
    try:
        slice_order = []
        for f in series_files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                num = int(getattr(ds, "InstanceNumber", 0))
            except Exception:
                num = 0
            slice_order.append((num, str(f)))
        slice_order.sort(key=lambda x: x[0])
        sorted_files = [x[1] for x in slice_order]

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_files)
        image = reader.Execute()
        sitk.WriteImage(image, str(output_path))
        log.info(f"    OK  Image: {output_path.name} | size={image.GetSize()}")
        return True
    except Exception as e:
        log.warning(f"    FAIL Image: {e}")
        return False


def pick_best_roi(rois: list) -> str:
    for keyword in TUMOR_ROI_KEYWORDS:
        for roi in rois:
            if keyword.lower() in roi.lower():
                return roi
    return rois[0]


def convert_rtstruct_to_mask(rtstruct_file: Path, dicom_files: list,
                              ref_nifti: Path, output_mask: Path) -> bool:
    """
    Convert RTStruct to binary NIfTI mask using pure pydicom rasterization.
    dicom_files: list of Path objects for the IMAGING series DICOM files.
    """
    try:
        rois = list_rt_structs_pydicom(rtstruct_file)
        if not rois:
            log.warning(f"    No ROIs found in: {rtstruct_file.name}")
            return False

        log.info(f"    ROIs available: {rois}")
        chosen_roi = pick_best_roi(rois)
        log.info(f"    Using ROI: '{chosen_roi}'")

        mask = rasterize_rtstruct(rtstruct_file, dicom_files, ref_nifti, chosen_roi)
        if mask is None:
            return False

        ref_img = nib.load(str(ref_nifti))
        out_img = nib.Nifti1Image(mask, ref_img.affine, ref_img.header)
        out_img.header.set_data_dtype(np.uint8)
        nib.save(out_img, str(output_mask))
        log.info(f"    OK  Mask saved: {output_mask.name}")
        return True

    except Exception as e:
        log.error(f"    FAIL Mask: {e}", exc_info=True)
        return False


def build_imaging_index(imaging_manifest_dir: Path) -> dict:
    log.info(f"Building imaging index from: {imaging_manifest_dir.name} ...")
    index = {}
    collection_dir = imaging_manifest_dir / "CPTAC-UCEC"
    if not collection_dir.exists():
        log.error(f"CPTAC-UCEC subfolder not found in {imaging_manifest_dir}")
        return index
    for patient_dir in sorted(collection_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        index[patient_id] = {}
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue
            index[patient_id][study_dir.name] = [
                s for s in study_dir.iterdir() if s.is_dir()
            ]
    log.info(f"  Indexed {len(index)} patients in imaging manifest.")
    return index


def find_series_in_index(index: dict, patient_id: str, study_name: str) -> list:
    if patient_id not in index:
        return []
    patient_studies = index[patient_id]
    if study_name in patient_studies:
        return list(patient_studies[study_name])
    date_part = study_name[:10]
    suffix    = study_name.rsplit("-", 1)[-1] if "-" in study_name else ""
    matched   = []
    for sname, slist in patient_studies.items():
        s_date   = sname[:10]
        s_suffix = sname.rsplit("-", 1)[-1] if "-" in sname else ""
        if s_date == date_part and s_suffix == suffix:
            matched.extend(slist)
    if matched:
        return matched
    for sname, slist in patient_studies.items():
        if sname[:10] == date_part:
            matched.extend(slist)
    return matched


def process_rtstruct_manifest(rt_manifest_dir: Path, rt_metadata_csv: Path,
                               imaging_index: dict, conversion_log: list):
    log.info(f"\nProcessing RTStruct manifest: {rt_manifest_dir.name}")

    df = pd.read_csv(rt_metadata_csv)
    df["Study Description"] = df["Study Description"].fillna("")
    df["inferred_modality"] = df["Study Description"].apply(infer_modality)
    df = df[~df["Series Description"].str.contains("SEED POINT", na=False)]
    if PRE_DOSE_ONLY:
        df = df[df["Series Description"].str.contains("Pre-Dose", na=False)]

    log.info(f"  RTStruct entries: {len(df)} | Patients: {df['Subject ID'].nunique()}")

    df["_rtstruct_folder"] = df["File Location"].apply(
        lambda loc: str(parse_windows_path(str(loc).strip(), rt_manifest_dir))
    )
    df["_study_name"] = df["_rtstruct_folder"].apply(lambda p: Path(p).parent.name)

    def roi_priority(desc):
        desc_lower = str(desc).lower()
        for i, kw in enumerate(TUMOR_ROI_KEYWORDS):
            if kw in desc_lower:
                return i
        return len(TUMOR_ROI_KEYWORDS)

    df["_roi_priority"] = df["Series Description"].apply(roi_priority)
    df = df.sort_values(["Subject ID", "_study_name", "_roi_priority"])
    best_rows = df.drop_duplicates(subset=["Subject ID", "_study_name"], keep="first")
    log.info(f"  Unique patient+study combinations: {len(best_rows)}")

    for _, row in tqdm(best_rows.iterrows(), total=len(best_rows), desc="  Converting"):
        patient_id  = str(row["Subject ID"]).strip()
        series_desc = str(row["Series Description"]).strip()
        modality    = str(row["inferred_modality"])
        file_loc    = str(row["File Location"]).strip()

        log.info(f"\n  Patient: {patient_id} | {modality} | {series_desc}")

        rtstruct_folder = parse_windows_path(file_loc, rt_manifest_dir)
        study_name      = rtstruct_folder.parent.name

        rt_files = [f for f in rtstruct_folder.iterdir()
                    if f.is_file()] if rtstruct_folder.exists() else []
        if not rt_files:
            log.warning(f"  RTStruct folder empty/missing: {rtstruct_folder}")
            conversion_log.append({
                "patient_id": patient_id, "modality": modality,
                "roi_desc": series_desc, "image_ok": False, "mask_ok": False,
                "reason": "rtstruct_file_not_found"
            })
            continue
        rtstruct_file = rt_files[0]

        series_folders = find_series_in_index(imaging_index, patient_id, study_name)
        if not series_folders:
            log.warning(f"  No imaging series found for {patient_id} / {study_name}")
            conversion_log.append({
                "patient_id": patient_id, "modality": modality,
                "roi_desc": series_desc, "image_ok": False, "mask_ok": False,
                "reason": "no_imaging_series_in_imaging_manifest"
            })
            continue

        # Collect imaging series, filter out non-image DICOM types
        imaging_series = []
        for sf in series_folders:
            dcm_files = [f for f in sf.iterdir() if f.is_file()]
            if not dcm_files:
                continue
            try:
                ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                s_mod = getattr(ds, "Modality", "")
                if s_mod in ("RTSTRUCT","RTDOSE","RTPLAN","SEG","SR","KO","PR","REG",""):
                    continue
                imaging_series.append({
                    "folder":      sf,
                    "modality":    s_mod,
                    "description": getattr(ds, "SeriesDescription", ""),
                    "files":       dcm_files,
                })
            except Exception as e:
                log.debug(f"  Could not read {dcm_files[0].name}: {e}")

        if not imaging_series:
            log.warning(f"  No image DICOMs in series folders")
            conversion_log.append({
                "patient_id": patient_id, "modality": modality,
                "roi_desc": series_desc, "image_ok": False, "mask_ok": False,
                "reason": "no_image_dicoms_in_series"
            })
            continue

        log.info(f"  Found {len(imaging_series)} imaging series | ROI: '{series_desc}'")

        study_date  = study_name.split("-")[0] if "-" in study_name else "unknown"
        patient_out = OUTPUT_DIR / modality / f"{patient_id}_{study_date}"
        patient_out.mkdir(parents=True, exist_ok=True)

        for i, series in enumerate(imaging_series):
            s_mod  = series["modality"]
            s_desc = series["description"].replace("/", "_").replace(" ", "_")[:40]
            s_name = f"{s_mod}_{s_desc}" if s_desc else f"{s_mod}_{i}"

            nifti_path = patient_out / f"{s_name}.nii.gz"
            mask_path  = patient_out / f"{s_name}_mask.nii.gz"

            log.info(f"  [{i+1}/{len(imaging_series)}] {s_mod} "
                     f"'{series['description']}' ({len(series['files'])} files)")

            if RESUME and nifti_path.exists() and mask_path.exists():
                log.info(f"  SKIP (already done): {nifti_path.name}")
                conversion_log.append({
                    "patient_id": patient_id, "modality": modality,
                    "series_modality": s_mod, "series_desc": series["description"],
                    "roi_desc": series_desc,
                    "nifti_path": str(nifti_path), "mask_path": str(mask_path),
                    "image_ok": True, "mask_ok": True, "reason": "skipped_already_done",
                })
                continue

            img_ok = convert_series_to_nifti(series["files"], nifti_path)
            if RESUME and nifti_path.exists() and not img_ok:
                img_ok = True

            mask_ok = False
            if img_ok:
                # Pass the actual DICOM file list directly - no wrong-folder risk
                mask_ok = convert_rtstruct_to_mask(
                    rtstruct_file, series["files"], nifti_path, mask_path)

            conversion_log.append({
                "patient_id":      patient_id,
                "modality":        modality,
                "series_modality": s_mod,
                "series_desc":     series["description"],
                "roi_desc":        series_desc,
                "nifti_path":      str(nifti_path) if img_ok else "",
                "mask_path":       str(mask_path) if mask_ok else "",
                "image_ok":        img_ok,
                "mask_ok":         mask_ok,
                "reason":          ("ok" if (img_ok and mask_ok) else
                                    "mask_failed" if img_ok else "image_failed"),
            })


def main():
    log.info("=" * 65)
    log.info("  CPTAC-UCEC DICOM -> NIfTI Converter v5 (pydicom rasterizer)")
    log.info("=" * 65)
    log.info(f"  Dataset root:  {DATASET_ROOT}")
    log.info(f"  Output dir:    {OUTPUT_DIR}")
    log.info(f"  Pre-dose only: {PRE_DOSE_ONLY}")
    log.info(f"  Resume:        {RESUME}")

    if not DATASET_ROOT.exists():
        log.error(f"Dataset root not found: {DATASET_ROOT}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("\nDiscovering NBIA manifest folders...")
    manifests = find_manifest_dirs(DATASET_ROOT)
    if len(manifests) < 2:
        log.error(f"Expected 2 manifest folders, found: {[m[0].name for m in manifests]}")
        return

    rt_manifest  = None
    img_manifest = None
    for mdir, mcsv in manifests:
        df_check = pd.read_csv(mcsv)
        col = df_check.get("Modality", df_check.get("SOP Class Name",
              pd.Series([""] * len(df_check))))
        rtstruct_count = col.astype(str).str.contains("RTSTRUCT|RT Structure", na=False).sum()
        log.info(f"  {mdir.name}: {len(df_check)} rows, ~{rtstruct_count} RTSTRUCT entries")
        if rtstruct_count > 10:
            rt_manifest = (mdir, mcsv)
        else:
            img_manifest = (mdir, mcsv)

    if rt_manifest is None or img_manifest is None:
        sorted_by_rows = sorted(manifests, key=lambda x: len(pd.read_csv(x[1])), reverse=True)
        rt_manifest, img_manifest = sorted_by_rows[0], sorted_by_rows[1]
        log.warning(f"  Fallback: RTStruct={rt_manifest[0].name}, Imaging={img_manifest[0].name}")

    log.info(f"\n  RTStruct manifest:  {rt_manifest[0].name}")
    log.info(f"  Imaging manifest:   {img_manifest[0].name}")

    imaging_index = build_imaging_index(img_manifest[0])
    if not imaging_index:
        log.error("Imaging index is empty.")
        return

    conversion_log = []
    process_rtstruct_manifest(rt_manifest[0], rt_manifest[1], imaging_index, conversion_log)

    log_df   = pd.DataFrame(conversion_log)
    log_path = OUTPUT_DIR / "conversion_log.csv"
    log_df.to_csv(log_path, index=False)

    total     = len(log_df)
    img_ok_n  = int(log_df["image_ok"].sum()) if total > 0 else 0
    mask_ok_n = int(log_df["mask_ok"].sum())  if total > 0 else 0

    log.info("\n" + "=" * 65)
    log.info("  DONE")
    log.info(f"  Total:     {total}")
    log.info(f"  Images OK: {img_ok_n}/{total}")
    log.info(f"  Masks OK:  {mask_ok_n}/{total}")
    if total > 0:
        for mod, grp in log_df.groupby("modality"):
            log.info(f"    {mod:<10s} | images={int(grp['image_ok'].sum())} "
                     f"masks={int(grp['mask_ok'].sum())} (n={len(grp)})")
    log.info(f"  Log: {log_path}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()