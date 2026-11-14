"""
Phase 8F-2: External Cross-Cohort Validation Suite (HER2 / Hematoxylin Cohort).
Generates a publication-grade teaser figure (3x6 Grid) for CVPR/IEEE double-column width (6.875 in).
Displays 3 representative tissue patches without Reinhard normalization, alongside evaluation metrics.
Includes a perfectly aligned vertical colorbar with scientific notation for predictive uncertainty.
"""

import sys
from pathlib import Path
from importlib import import_module
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# =========================================================================
# PATH CONFIGURATION
# =========================================================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "code" else CURRENT_FILE.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CODE_DIR = PROJECT_ROOT / "code"
if CODE_DIR.exists():
    sys.path.append(str(CODE_DIR))

STUDENT_WEIGHTS = RESULTS_DIR / "best_student.pth"
if not STUDENT_WEIGHTS.exists():
    STUDENT_WEIGHTS = Path("/workspace/results/best_student.pth")

TEST3_DIR = PROJECT_ROOT / "dataset" / "test3"
if not TEST3_DIR.exists():
    TEST3_DIR = Path("/workspace/dataset/test3")

CACHE_DIR = PROJECT_ROOT / "dataset" / "test3_patches_100"
if not CACHE_DIR.exists():
    CACHE_DIR = Path("/workspace/dataset/test3_patches_100")

# =========================================================================
# PUBLICATION TYPOGRAPHY CONFIGURATION (CVPR / IEEE SPECS)
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  
    "ps.fonttype": 42,
})


def extract_and_cache_test3_patches(source_dir: Path, cache_dir: Path, target_count: int = 100, patch_size: int = 512):
    """Extracts and caches tissue patches from whole slide images."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing_patches = sorted(list(cache_dir.glob("*.png")))

    if len(existing_patches) >= target_count:
        print(f"[*] Found {len(existing_patches)} cached patches in {cache_dir}. Loading {target_count}...")
        return existing_patches[:target_count]

    print(f"[*] Extracting new patches from {source_dir} (Target: {target_count})...")
    for p in existing_patches:
        p.unlink()

    wsi_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        wsi_files.extend(list(source_dir.rglob(ext)))

    extracted_paths = []
    count = 0
    step = patch_size

    for slide_path in sorted(wsi_files):
        if count >= target_count:
            break
        try:
            img = Image.open(slide_path).convert("RGB")
            img_np = np.array(img)
            h, w, _ = img_np.shape

            for y in range(0, h - patch_size + 1, step):
                for x in range(0, w - patch_size + 1, step):
                    if count >= target_count:
                        break
                    patch = img_np[y:y+patch_size, x:x+patch_size, :]
                    gray = np.mean(patch, axis=2) / 255.0

                    # Tissue presence threshold (filter out mostly white background)
                    if np.mean(gray < 0.92) > 0.35:
                        out_name = cache_dir / f"her2_patch_{count:03d}.png"
                        Image.fromarray(patch).save(out_name)
                        extracted_paths.append(out_name)
                        count += 1
        except Exception:
            pass

    return extracted_paths


def run_inference(model, physics, img_np: np.ndarray, device: torch.device):
    """Runs a single patch through the student model and applies the physical forward model."""
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        c_mean, u_var, _ = model.forward_train(img_tensor)
        recon_tensor = physics.conc_to_rgb(c_mean)
        stains = model.get_positive_stains(img_tensor)

    return (
        recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        stains[0, 0].cpu().numpy(),
        stains[0, 1].cpu().numpy(),
        stains[0, 2].cpu().numpy(),
        u_var.squeeze(0).mean(dim=0).cpu().numpy()
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== External Validation (Test3 HER2 Cohort - 100 Patches) ===")

    try:
        FastStudentUNet = import_module("studentDistillation").FastStudentUNet
        BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics
    except ModuleNotFoundError as e:
        print(f"[!] Modules not found: {e}. Check PYTHONPATH.")
        return

    if not STUDENT_WEIGHTS.exists():
        print(f"[!] Checkpoint not found at: {STUDENT_WEIGHTS}. Aborting.")
        return

    # Load Model & Physics module
    model = FastStudentUNet().to(device)
    model.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    physics = BeerLambertPhysics().to(device)

    patch_paths = extract_and_cache_test3_patches(TEST3_DIR, CACHE_DIR, 100)
    specimen_records = []
    first_three_patches = []

    for idx, p_path in enumerate(tqdm(patch_paths, desc="Evaluating HER2 100 Patches")):
        orig_np = np.array(Image.open(p_path).convert("RGB"), dtype=np.float32) / 255.0

        recon, c1, c2, c3, u = run_inference(model, physics, orig_np, device)
        s1, s2, s3 = float(np.sum(c1)), float(np.sum(c2)), float(np.sum(c3))
        total_stain = s1 + s2 + s3 + 1e-8

        specimen_records.append({
            "Patch_ID": idx,
            "HER2_Dense_Pct": (s1 / (s1 + s2 + 1e-8)) * 100.0,
            "HER2_Membrane_Pct": (s2 / (s1 + s2 + 1e-8)) * 100.0,
            "Hematoxylin_Pct": (s3 / total_stain) * 100.0,
            "Uncertainty": float(np.mean(u))
        })

        # Cache data for the first 3 representative patches for visualization
        if idx < 3:
            first_three_patches.append({
                "patch_id": idx,
                "orig": orig_np,
                "recon": recon,
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "u": u
            })

    # =========================================================================
    # RENDER UNIFIED 3-PATCH PUBLICATION FIGURE (3 Rows x 6 Columns)
    # =========================================================================
    if first_three_patches:
        # Full 2-column width: 6.875 in x 3.4 in
        fig, axes = plt.subplots(3, 6, figsize=(6.875, 3.4), dpi=300)
        
        # Adjusted right margin (0.90) to prevent overlap with the vertical colorbar
        plt.subplots_adjust(wspace=0.04, hspace=0.10, top=0.88, bottom=0.03, left=0.08, right=0.90)

        col_headers = [
            "Input\nRGB",
            "Reconstruction\n(Physics)",
            "HER2\nDense",
            "HER2\nMembrane",
            "Counterstain\nHematoxylin",
            "Predictive\nUncertainty"
        ]

        # Calculate shared color limits for fair intensity comparison
        v1 = max([float(np.percentile(p["c1"], 99.2)) for p in first_three_patches] + [0.08])
        v2 = max([float(np.percentile(p["c2"], 99.2)) for p in first_three_patches] + [0.08])
        v3 = max([float(np.percentile(p["c3"], 99.2)) for p in first_three_patches] + [0.08])
        u_max = max([float(np.percentile(p["u"], 99.0)) for p in first_three_patches] + [1e-4])

        for row_idx, p_data in enumerate(first_three_patches):
            row_items = [
                (np.clip(p_data["orig"], 0, 1), None, None),
                (np.clip(p_data["recon"], 0, 1), None, None),
                (p_data["c1"], "inferno", (0, v1)),
                (p_data["c2"], "inferno", (0, v2)),
                (p_data["c3"], "inferno", (0, v3)),
                (p_data["u"], "viridis", (0, u_max))
            ]

            for col_idx, (img, cmap, clim) in enumerate(row_items):
                ax = axes[row_idx, col_idx]

                if cmap is None:
                    ax.imshow(img)
                else:
                    vmin, vmax = clim
                    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

                # Column headers only on the top row
                if row_idx == 0:
                    ax.set_title(col_headers[col_idx], fontsize=5.5, fontweight="bold", pad=3, linespacing=1.05)

                # Row label showing patch identifier
                if col_idx == 0:
                    ax.set_ylabel(f"Patch #{p_data['patch_id'] + 1}", fontsize=6.0, fontweight="bold", labelpad=3)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis("off")

        # --- Vertical Colorbar for Uncertainty ---
        # Height perfectly aligned with the grid: bottom=0.03, height=0.88 - 0.03 = 0.85
        cbar_ax = fig.add_axes([0.91, 0.03, 0.015, 0.85]) 
        
        # Map colors based on 0 to max uncertainty
        norm = mcolors.Normalize(vmin=0, vmax=u_max)
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        
        # Draw Colorbar (Title/label removed)
        cbar = fig.colorbar(sm, cax=cbar_ax)
        
        # Setup scientific notation (IEEE/CVPR standard math text format e.g., 10^-3)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(formatter)
        
        # Increase tick label and offset text (e.g., 10^-4) font sizes for better readability
        cbar.ax.tick_params(labelsize=8, length=3, pad=2)
        cbar.ax.yaxis.get_offset_text().set_fontsize(8)
        cbar.outline.set_linewidth(0.5)
        # ----------------------------------------

        combined_pdf_path = RESULTS_DIR / "HER2_External_validation.pdf"
        plt.savefig(combined_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close()
        print(f"[✓] Saved unified 3-patch figure to: {combined_pdf_path}")

    # =========================================================================
    # METRICS SUMMARY & CSV EXPORT
    # =========================================================================
    df_out = pd.DataFrame(specimen_records)
    csv_path = RESULTS_DIR / "external_test3_metrics.csv"
    df_out.to_csv(csv_path, index=False)

    print("\n=== HER2 100-Patch Summary (Mean ± Std) ===")
    print(f"Dense HER2:          {df_out['HER2_Dense_Pct'].mean():.2f}% ± {df_out['HER2_Dense_Pct'].std():.2f}%")
    print(f"Membrane HER2:       {df_out['HER2_Membrane_Pct'].mean():.2f}% ± {df_out['HER2_Membrane_Pct'].std():.2f}%")
    print(f"Hematoxylin:         {df_out['Hematoxylin_Pct'].mean():.2f}% ± {df_out['Hematoxylin_Pct'].std():.2f}%")
    print(f"Uncertainty:         {df_out['Uncertainty'].mean():.6f} ± {df_out['Uncertainty'].std():.6f}")
    print(f"[✓] Saved metrics to: {csv_path}")


if __name__ == "__main__":
    main()