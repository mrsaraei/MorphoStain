"""
Level-0 High-Resolution Patch Tiling & Extraction Pipeline.
Extracts non-overlapping 512x512 patches from whole slide images (WSI)
and normalizes background illumination artifacts.
"""
import os
import shutil
import cv2
import numpy as np
import openslide
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# Configuration & Hyperparameters
# ==========================================
DATA_DIR = "/workspace/data"        # Standardized input directory containing downloaded WSIs
PATCH_DIR = "/workspace/patch"      # Output directory for extracted level-0 patches
PATCH_SIZE = 512                   # Standardized evaluation patch size
TISSUE_THRESHOLD = 0.3             # Required foreground tissue occupancy
MIN_VALID_PATCHES = 50             # Slide directory pruning threshold
NORMALIZE_BACKGROUND = True        # Flat-field background illumination calibration
NUM_THREADS = os.cpu_count() or 4  # Concurrent worker thread capacity


def estimate_background_intensity(slide, downsample_factor=32):
    """Estimates slide background white-point reference (I_0)."""
    w, h = slide.dimensions
    thumb = slide.get_thumbnail((w // downsample_factor, h // downsample_factor))
    thumb_arr = np.array(thumb)[..., :3].astype(np.float32)

    hsv = cv2.cvtColor(thumb_arr.astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    bright_cutoff = np.percentile(value, 85)
    glass_mask = (saturation < 10) & (value >= bright_cutoff)

    if np.sum(glass_mask) < 100:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)

    return thumb_arr[glass_mask].mean(axis=0)


def generate_tissue_mask(slide, bg_intensity, downsample_factor=32):
    """Generates a macroscopic binary mask to index tissue locations."""
    w, h = slide.dimensions
    thumb = slide.get_thumbnail((w // downsample_factor, h // downsample_factor))
    thumb_arr = np.array(thumb)[..., :3].astype(np.float32)

    corrected = np.clip((thumb_arr / (bg_intensity + 1e-5)) * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(corrected, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:, :, 1]

    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    tissue_mask = (otsu_mask > 0) | (s_channel > 6)

    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def process_single_patch(slide, x, y, patch_path, bg_intensity):
    """Extracts, validates tissue coverage, and writes a single patch."""
    patch_rgba = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
    patch_rgb = np.array(patch_rgba)[..., :3].astype(np.float32)

    if NORMALIZE_BACKGROUND:
        corrected_rgb = np.clip((patch_rgb / (bg_intensity + 1e-5)) * 255.0, 0, 255).astype(np.uint8)
    else:
        corrected_rgb = patch_rgb.astype(np.uint8)

    patch_gray = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2GRAY)
    patch_hsv = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2HSV)

    patch_tissue_mask = (patch_gray < 235) & ((patch_hsv[:, :, 1] > 5) | (patch_gray < 200))

    if np.mean(patch_tissue_mask) >= TISSUE_THRESHOLD:
        patch_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(patch_path, patch_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return 1

    return 0


def process_slide(svs_path):
    """Executes Level-0 patch extraction pipeline for a single WSI."""
    slide_name = os.path.splitext(os.path.basename(svs_path))[0]
    slide_patch_dir = os.path.join(PATCH_DIR, slide_name)
    os.makedirs(slide_patch_dir, exist_ok=True)

    try:
        slide = openslide.OpenSlide(svs_path)
    except Exception as e:
        print(f"\n[Error] Failed to load {slide_name}: {e}")
        return 0

    width, height = slide.dimensions
    cols = width // PATCH_SIZE
    rows = height // PATCH_SIZE

    bg_intensity = estimate_background_intensity(slide)
    print(f"\n=> Processing Slide: {slide_name} | Dims: {width}x{height} | Background Ref: {np.round(bg_intensity, 1)}")

    downsample_factor = 32
    mask = generate_tissue_mask(slide, bg_intensity, downsample_factor)

    existing_patches = set(os.listdir(slide_patch_dir))
    saved_count = len(existing_patches)

    valid_tasks = []
    for row in range(rows):
        for col in range(cols):
            x = col * PATCH_SIZE
            y = row * PATCH_SIZE
            patch_filename = f"{x}_{y}.jpg"

            if patch_filename in existing_patches:
                continue

            mask_x = x // downsample_factor
            mask_y = y // downsample_factor
            mask_s = max(1, PATCH_SIZE // downsample_factor)

            mask_region = mask[mask_y:mask_y+mask_s, mask_x:mask_x+mask_s]
            if np.mean(mask_region) >= TISSUE_THRESHOLD:
                patch_path = os.path.join(slide_patch_dir, patch_filename)
                valid_tasks.append((x, y, patch_path))

    if valid_tasks:
        new_patches = 0
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = {
                executor.submit(process_single_patch, slide, task[0], task[1], task[2], bg_intensity): task
                for task in valid_tasks
            }
            with tqdm(total=len(futures), desc="Extracting Patches", leave=False) as pbar:
                for future in as_completed(futures):
                    new_patches += future.result()
                    pbar.update(1)
        saved_count += new_patches

    slide.close()

    if saved_count < MIN_VALID_PATCHES:
        if os.path.exists(slide_patch_dir):
            shutil.rmtree(slide_patch_dir)
        print(f"[REMOVED] {slide_name} pruned (insufficient tissue yield: {saved_count} patches).")
        return 0

    return saved_count


def main():
    os.makedirs(PATCH_DIR, exist_ok=True)
    svs_files = glob(os.path.join(DATA_DIR, "**", "*.svs"), recursive=True)

    if not svs_files:
        print("No .svs files found in the data directory!")
        return

    print(f"Found {len(svs_files)} .svs file(s) for processing.")
    results = {}

    for svs_path in svs_files:
        count = process_slide(svs_path)
        results[os.path.basename(svs_path)] = count

    print("\n" + "="*50)
    print("Final Patch Extraction Report:")
    print("="*50)
    for name, count in results.items():
        if count >= MIN_VALID_PATCHES:
            print(f"[SUCCESS] Slide {name}: {count} patches extracted and retained.")
        else:
            print(f"[DISCARDED] Slide {name}: Excluded due to insufficient patches.")
    print("="*50)


if __name__ == "__main__":
    main()