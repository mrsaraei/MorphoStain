"""
Data Ingestion Pipeline for MorphoStain.
Streams high-resolution whole slide images (WSI) from the SEA-AD S3 bucket
directly to the standardized container workspace path (/workspace/data).
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

# =============================================================================
# 1. Configuration & Constants
# =============================================================================
DATA_DIR = Path("/workspace/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BUCKET_NAME = "sea-ad-quantitative-neuropathology"
PREFIX = "middle-temporal-gyrus/"

STAINS_TO_FETCH: Dict[str, str] = {
    "AT": "Tau (AT8)",
    "I6": "Amyloid-Beta (6E10)",
}

MIN_TISSUE_PERCENTAGE = 1.0   
MIN_BLUR_THRESHOLD = 50.0     


# =============================================================================
# 2. S3 Slide Ingestion Functions
# =============================================================================
def get_s3_client():
    """Initializes and returns an anonymous S3 client for public buckets."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def fetch_donor_list(s3_client, max_donors: int = 5) -> List[str]:
    """Queries S3 directory prefixes to retrieve valid donor identifiers."""
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME, Prefix=PREFIX, Delimiter="/"
    )

    donors = []
    for common_prefix in response.get("CommonPrefixes", []):
        folder_name = common_prefix["Prefix"].rstrip("/").split("/")[-1]
        if folder_name.startswith("H"):
            donors.append(folder_name)

    return donors[:max_donors]


def download_svs_slide(
    s3_client,
    donor: str,
    stain_key: str,
    stain_label: str,
    target_dir: Path
) -> Optional[Path]:
    """Locates and downloads the first .svs slide for a specific donor and stain."""
    donor_prefix = f"{PREFIX}{donor}/"
    subdirs_resp = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME, Prefix=donor_prefix, Delimiter="/"
    )

    stain_prefix = None
    for item in subdirs_resp.get("CommonPrefixes", []):
        if stain_key in item["Prefix"]:
            stain_prefix = item["Prefix"]
            break

    if not stain_prefix:
        print(f"  [-] Stain folder missing: {stain_label} ({stain_key}) for {donor}")
        return None

    svs_resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=stain_prefix)
    for obj in svs_resp.get("Contents", []):
        s3_key = obj["Key"]
        if s3_key.endswith(".svs"):
            filename = Path(s3_key).name
            local_path = target_dir / filename

            if local_path.exists():
                print(f"  [=] Exists locally: {filename}")
            else:
                print(f"  [+] Downloading {stain_label}: {filename}...")
                s3_client.download_file(BUCKET_NAME, s3_key, str(local_path))
            return local_path

    return None


# =============================================================================
# 3. Whole Slide Image (WSI) Quality Assessment
# =============================================================================
def extract_slide_metadata(slide: openslide.OpenSlide, slide_path: Path) -> pd.DataFrame:
    """Extracts physical acquisition scale and scanner metadata."""
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "Unknown")
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, "Unknown")
    vendor = slide.properties.get(openslide.PROPERTY_NAME_VENDOR, "Unknown")
    objective = slide.properties.get("openslide.objective-power", "Unknown")

    return pd.DataFrame([{
        "Slide Name": slide_path.name,
        "Base Dimensions (px)": f"{slide.dimensions[0]} x {slide.dimensions[1]}",
        "Pyramid Levels": slide.level_count,
        "Scanner Vendor": vendor,
        "Objective Magnification": f"{objective}x" if objective != "Unknown" else objective,
        "MPP X (um/px)": mpp_x,
        "MPP Y (um/px)": mpp_y
    }])


def segment_tissue_and_evaluate_qc(
    slide: openslide.OpenSlide,
    downsample_level: int = 2
) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
    """Evaluates tissue presence via saturation Otsu thresholding and Laplacian variance."""
    actual_level = min(downsample_level, slide.level_count - 1)
    level_dims = slide.level_dimensions[actual_level]
    
    thumb_pil = slide.read_region((0, 0), actual_level, level_dims).convert("RGB")
    thumb_rgb = np.array(thumb_pil)

    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    
    _, tissue_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    tissue_area_px = np.count_nonzero(tissue_mask)
    tissue_percent = (tissue_area_px / tissue_mask.size) * 100.0

    gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    blur_score = float(np.var(laplacian[tissue_mask > 0])) if tissue_area_px > 0 else 0.0
    qc_passed = (tissue_percent >= MIN_TISSUE_PERCENTAGE) and (blur_score >= MIN_BLUR_THRESHOLD)

    return thumb_rgb, tissue_mask, tissue_percent, blur_score, qc_passed


# =============================================================================
# 4. Pipeline Execution
# =============================================================================
def main():
    s3 = get_s3_client()
    print(f"Target download directory set to: {DATA_DIR.resolve()}\n")

    print("Fetching donor candidates from S3...")
    selected_donors = fetch_donor_list(s3, max_donors=5)
    print(f"Selected {len(selected_donors)} donors: {selected_donors}\n")

    print("Beginning slide ingestion...")
    for idx, donor in enumerate(selected_donors, 1):
        print(f"[{idx}/{len(selected_donors)}] Processing Donor: {donor}")
        for stain_key, stain_label in STAINS_TO_FETCH.items():
            download_svs_slide(s3, donor, stain_key, stain_label, DATA_DIR)
        print("-" * 50)

    downloaded_slides = sorted(list(DATA_DIR.glob("*.svs")))
    if not downloaded_slides:
        print("No .svs files found in target directory. Terminating.")
        sys.exit(0)

    sample_slide_path = downloaded_slides[0]
    print(f"\nProcessing WSI Quality Control on: {sample_slide_path.name}")
    print("=" * 60)

    with openslide.OpenSlide(str(sample_slide_path)) as slide:
        metadata_df = extract_slide_metadata(slide, sample_slide_path)
        print(metadata_df.T.to_string(header=False))
        print("-" * 60)

        thumb_rgb, mask, tissue_pct, blur_val, passed = segment_tissue_and_evaluate_qc(slide)

    print(f"Tissue Coverage : {tissue_pct:.2f}% (Threshold >= {MIN_TISSUE_PERCENTAGE}%)")
    print(f"Focus Sharpness : {blur_val:.2f} (Threshold >= {MIN_BLUR_THRESHOLD})")
    print(f"QC Status       : {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()