"""
Phase 8D: Pixel-Level Reconstruction Error Map Generator.
Renders publication-grade 3-panel diagnostic figures matching CVPR/IEEE physical column widths.
Displays: (Original RGB, MorphoStain Reconstructed RGB, and Calibrated Pixel-Wise MSE Heatmap).
Saves to: RESULTS_DIR / Fig_Error_Map_Patch_{0..4}.pdf
"""

import sys
from pathlib import Path
from importlib import import_module
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

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

TEST_H5_PATH = PROJECT_ROOT / "latent_test.h5"
if not TEST_H5_PATH.exists():
    TEST_H5_PATH = Path("/workspace/latent_test.h5")

# =========================================================================
# PUBLICATION TYPOGRAPHY CONFIGURATION (CVPR / IEEE SPECS)
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 6,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # Mandatory Type 42 TrueType fonts for CVPR/IEEE
    "ps.fonttype": 42,
})


def main():
    print("=== Generating High-Resolution Pixel-Level Error Maps (CVPR Format) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        FastStudentUNet = import_module("studentDistillation").FastStudentUNet
        BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics
    except ModuleNotFoundError as e:
        print(f"[!] Modules not found: {e}. Check PYTHONPATH.")
        return

    if not STUDENT_WEIGHTS.exists():
        print(f"[!] Checkpoint not found at: {STUDENT_WEIGHTS}. Aborting.")
        return

    student = FastStudentUNet().to(device)
    student.load_state_dict(
        torch.load(STUDENT_WEIGHTS, map_location=device, weights_only=True)
    )
    student.eval()
    physics = BeerLambertPhysics().to(device)

    if not TEST_H5_PATH.exists():
        print(f"[!] Test H5 file not found at: {TEST_H5_PATH}. Aborting.")
        return

    with h5py.File(TEST_H5_PATH, "r") as f:
        N = f["rgb"].shape[0]
        patch_indices = [0, min(1, N - 1), min(2, N - 1), min(3, N - 1), min(4, N - 1)]

        for out_idx, p_idx in enumerate(patch_indices):
            rgb_np = f["rgb"][p_idx]
            rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(device)

            with torch.no_grad():
                c_mean, _ = student(rgb_t)
                recon_t = physics.conc_to_rgb(c_mean)

                # Pixel-wise MSE across RGB channels
                mse_map = torch.mean((recon_t - rgb_t) ** 2, dim=1).squeeze(0).cpu().numpy()
                recon_np = recon_t.squeeze(0).cpu().numpy()

            orig_disp = np.clip(np.transpose(rgb_np, (1, 2, 0)), 0.0, 1.0)
            recon_disp = np.clip(np.transpose(recon_np, (1, 2, 0)), 0.0, 1.0)

            # CVPR full 2-column or wide sub-figure: 6.875 in x 2.2 in
            fig, axes = plt.subplots(
                1, 3, figsize=(6.875, 2.2), dpi=300,
                gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]}
            )
            plt.subplots_adjust(wspace=0.08, left=0.02, right=0.94, top=0.88, bottom=0.04)

            # Panel 1: Original RGB
            axes[0].imshow(orig_disp)
            axes[0].set_title("Input RGB", fontweight="bold", pad=4)
            axes[0].axis("off")

            # Panel 2: MorphoStain Recon
            axes[1].imshow(recon_disp)
            axes[1].set_title("Physics Reconstruction", fontweight="bold", pad=4)
            axes[1].axis("off")

            # Panel 3: Calibrated Error Map
            im_err = axes[2].imshow(mse_map, cmap="magma", vmin=0.000, vmax=0.005)
            axes[2].set_title("Reconstruction Error (MSE)", fontweight="bold", pad=4)
            axes[2].axis("off")

            # Colorbar configuration
            cbar = fig.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.03)
            cbar.set_ticks([0.000, 0.001, 0.002, 0.003, 0.004, 0.005])
            cbar.set_ticklabels(["0.000", "0.001", "0.002", "0.003", "0.004", "0.005"])
            cbar.ax.tick_params(labelsize=5, length=2, pad=1)

            save_path = RESULTS_DIR / f"Fig_Error_Map_Patch_{out_idx}.pdf"
            plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
            plt.close()
            print(f"  [✓] Saved: {save_path} (Mean MSE: {mse_map.mean():.6f})")


if __name__ == "__main__":
    main()