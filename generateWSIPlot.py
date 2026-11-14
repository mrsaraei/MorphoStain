"""
Phase 8A: True Whole-Slide Image (WSI) Stitching & Macro-Reconstruction (5-Panel).
Reads individual patches from a specific slide, parses X/Y coordinates from filenames,
and stitches them back into a full macroscopic WSI grid.
Generates publication-standard PDF figures formatted for IEEE/CVPR layouts.
"""
import os
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from importlib import import_module

# =========================================================================
# PUBLICATION TYPOGRAPHY & MATPLOTLIB CONFIGURATION (CVPR/IEEE Standard)
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # TrueType fonts compliance for CVPR/IEEE
    "ps.fonttype": 42,
})

def norm_relative(arr: np.ndarray, ref_max: float) -> np.ndarray:
    if ref_max < 1e-4:
        return np.zeros_like(arr)
    return np.clip(arr / ref_max, 0.0, 1.0)

def reconstruct_full_slide(student, physics, device, slide_prefix, stain_mode, save_path, downscale_factor=4):
    print(f"\n--- Processing Full Slide: {slide_prefix} ({stain_mode}) ---")
    patch_files = glob.glob(f"/workspace/dataset/test/{slide_prefix}_*.jpg")
    
    if not patch_files:
        print(f"No patches found for {slide_prefix} in test set.")
        return

    coords = []
    for f in patch_files:
        basename = os.path.basename(f).replace(".jpg", "")
        parts = basename.split("_")
        x, y = int(parts[-2]), int(parts[-1])
        coords.append((x, y, f))
    
    min_x = min(c[0] for c in coords)
    max_x = max(c[0] for c in coords)
    min_y = min(c[1] for c in coords)
    max_y = max(c[1] for c in coords)
    
    orig_patch_size = 512
    scaled_patch_size = orig_patch_size // downscale_factor
    
    grid_w = (max_x - min_x + orig_patch_size) // downscale_factor
    grid_h = (max_y - min_y + orig_patch_size) // downscale_factor

    print(f"Stitching {len(patch_files)} patches into a {grid_w}x{grid_h} WSI Canvas...")

    # Create empty canvases including Hematoxylin
    canvas_orig = np.ones((grid_h, grid_w, 3), dtype=np.float32)
    canvas_recon = np.ones((grid_h, grid_w, 3), dtype=np.float32)
    canvas_dab = np.zeros((grid_h, grid_w), dtype=np.float32)
    canvas_htx = np.zeros((grid_h, grid_w), dtype=np.float32)
    canvas_unc = np.zeros((grid_h, grid_w), dtype=np.float32)

    with torch.no_grad():
        for x, y, fpath in coords:
            img = Image.open(fpath).convert("RGB")
            rgb_np = np.array(img, dtype=np.float32) / 255.0
            rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device)

            c_mean, u_var, attn_prob = student.forward_train(rgb_t)
            recon_t = physics.conc_to_rgb(c_mean)

            emp_err = torch.mean((recon_t - rgb_t) ** 2, dim=1).squeeze(0)
            u_np = u_var.mean(dim=1).squeeze(0)
            combined_u = 0.60 * u_np + 0.40 * emp_err

            stains = student.get_positive_stains(rgb_t)
            c_dab_pure = stains[0, 0] + stains[0, 1]
            c_htx = stains[0, 2]

            attn_raw = attn_prob[0, 0]
            if stain_mode == "Tau":
                routed = c_dab_pure * (1.0 - attn_raw)
            else:
                routed = c_dab_pure * attn_raw

            rgb_small = F.interpolate(rgb_t, size=(scaled_patch_size, scaled_patch_size), mode='area').squeeze().permute(1, 2, 0).cpu().numpy()
            recon_small = F.interpolate(recon_t, size=(scaled_patch_size, scaled_patch_size), mode='area').squeeze().permute(1, 2, 0).cpu().numpy()
            routed_small = F.interpolate(routed.unsqueeze(0).unsqueeze(0), size=(scaled_patch_size, scaled_patch_size), mode='area').squeeze().cpu().numpy()
            htx_small = F.interpolate(c_htx.unsqueeze(0).unsqueeze(0), size=(scaled_patch_size, scaled_patch_size), mode='area').squeeze().cpu().numpy()
            unc_small = F.interpolate(combined_u.unsqueeze(0).unsqueeze(0), size=(scaled_patch_size, scaled_patch_size), mode='area').squeeze().cpu().numpy()

            px_x = (x - min_x) // downscale_factor
            px_y = (y - min_y) // downscale_factor

            canvas_orig[px_y:px_y+scaled_patch_size, px_x:px_x+scaled_patch_size] = rgb_small
            canvas_recon[px_y:px_y+scaled_patch_size, px_x:px_x+scaled_patch_size] = recon_small
            canvas_dab[px_y:px_y+scaled_patch_size, px_x:px_x+scaled_patch_size] = routed_small
            canvas_htx[px_y:px_y+scaled_patch_size, px_x:px_x+scaled_patch_size] = htx_small
            canvas_unc[px_y:px_y+scaled_patch_size, px_x:px_x+scaled_patch_size] = unc_small

    canvas_orig = np.clip(canvas_orig, 0.0, 1.0)
    canvas_recon = np.clip(canvas_recon, 0.0, 1.0)
    
    canvas_unc = gaussian_filter(canvas_unc, sigma=2.0)
    p99_u = float(np.percentile(canvas_unc[canvas_unc > 1e-6], 99.2)) if np.any(canvas_unc > 1e-6) else 1.0
    canvas_unc_scaled = np.clip((canvas_unc / p99_u) * 0.022, 0.0, 0.025)

    canvas_dab = gaussian_filter(canvas_dab, sigma=1.0)
    dab_peak = float(np.percentile(canvas_dab[canvas_dab > 1e-6], 99.2)) if np.any(canvas_dab > 1e-6) else 1.0
    canvas_dab_vis = norm_relative(canvas_dab, dab_peak)

    canvas_htx = gaussian_filter(canvas_htx, sigma=1.0)
    htx_peak = float(np.percentile(canvas_htx[canvas_htx > 1e-6], 99.2)) if np.any(canvas_htx > 1e-6) else 1.0
    canvas_htx_vis = norm_relative(canvas_htx, htx_peak)

    # 5-Panel WSI Layout (Width adjusted to 7.0 inches for double-column IEEE/CVPR layout)
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.4), dpi=300)
    plt.subplots_adjust(wspace=0.04, hspace=0.0, top=0.88, bottom=0.02, left=0.01, right=0.99)
    
    axes[0].imshow(canvas_orig)
    axes[0].set_title(f"Original WSI\n({stain_mode})", fontweight="bold", pad=3)
    axes[0].axis("off")

    axes[1].imshow(canvas_recon)
    axes[1].set_title("MorphoStain\nRecon", fontweight="bold", pad=3)
    axes[1].axis("off")

    channel_name = r"Unmixed Tau (AT8)" if stain_mode == "Tau" else r"Unmixed A$\beta$ (6E10)"
    axes[2].imshow(canvas_dab_vis, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"{channel_name}\n[Physical DAB]", fontweight="bold", pad=3)
    axes[2].axis("off")

    axes[3].imshow(canvas_htx_vis, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[3].set_title("Hematoxylin\n[Counterstain]", fontweight="bold", pad=3)
    axes[3].axis("off")

    im_u = axes[4].imshow(canvas_unc_scaled, cmap="inferno", vmin=0.000, vmax=0.025)
    axes[4].set_title("Spatial Uncertainty\nMap", fontweight="bold", pad=3)
    axes[4].axis("off")

    cbar = fig.colorbar(im_u, ax=axes[4], fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=5)

    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"  [✓] Full WSI Stitch saved to {save_path}")

def main():
    print("=== Phase 8A: True Macro WSI Stitching (with Hematoxylin) ===")
    os.makedirs("/workspace/results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FastStudentUNet = import_module("studentDistillation").FastStudentUNet
    BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics

    student = FastStudentUNet().to(device)
    student.load_state_dict(torch.load("/workspace/results/best_student.pth", map_location=device, weights_only=True))
    student.eval()
    physics = BeerLambertPhysics().to(device)

    tau_slide_id = "H20.33.004-A6-AT"
    abeta_slide_id = "H20.33.004-A6-I6"

    tau_path = "/workspace/results/WSI_Tau_Reconstruction.pdf"
    reconstruct_full_slide(student, physics, device, tau_slide_id, "Tau", tau_path)

    abeta_path = "/workspace/results/WSI_Amyloid_Reconstruction.pdf"
    reconstruct_full_slide(student, physics, device, abeta_slide_id, "Amyloid", abeta_path)

if __name__ == "__main__":
    main()