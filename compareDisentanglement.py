"""
Phase 6D: Biomarker Disentanglement & Leakage Comparison (MorphoStain vs. InverseAE).
Generates the publication teaser figure (Disentanglement_Comparison.pdf) matching
the exact physical column width of 2-column LaTeX templates (CVPR/IEEE).
"""

import os
import sys
from pathlib import Path
from importlib import import_module
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =========================================================================
# PATH CONFIGURATION
# =========================================================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "code" else CURRENT_FILE.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

H5_PATH = PROJECT_ROOT / "latent_test.h5"
if not H5_PATH.exists():
    H5_PATH = Path("/workspace/latent_test.h5")

STUDENT_WEIGHTS = RESULTS_DIR / "best_student.pth"
if not STUDENT_WEIGHTS.exists():
    STUDENT_WEIGHTS = Path("/workspace/results/best_student.pth")

INVERSE_AE_WEIGHTS = RESULTS_DIR / "best_inverse_ae.pth"
if not INVERSE_AE_WEIGHTS.exists():
    INVERSE_AE_WEIGHTS = Path("/workspace/results/best_inverse_ae.pth")

# Add code directory to path to ensure imports work
CODE_DIR = PROJECT_ROOT / "code"
if CODE_DIR.exists():
    sys.path.append(str(CODE_DIR))
elif str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import the correct updated InverseAE architecture directly from the training script
from trainInverseAE import InverseAEModel

# =========================================================================
# PUBLICATION TYPOGRAPHY CONFIGURATION
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def norm_calibrated(arr: np.ndarray, ref_max: float) -> np.ndarray:
    """Normalizes the array based on a calibrated maximum value."""
    ref_max = max(ref_max, 0.15)
    return np.clip(arr / ref_max, 0.0, 1.0)


def main():
    print("=== Generating Calibrated Biomarker Disentanglement Comparison (Teaser) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dynamically import student architecture
    try:
        FastStudentUNet = import_module("studentDistillation").FastStudentUNet
    except ModuleNotFoundError as e:
        print(f"[!] Import failed: {e}. Check PYTHONPATH.")
        return

    if not STUDENT_WEIGHTS.exists() or not INVERSE_AE_WEIGHTS.exists():
        print(f"[!] Checkpoints not found in {RESULTS_DIR}. Aborting.")
        return

    # Load MorphoStain (Student) model
    student = FastStudentUNet().to(device)
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device, weights_only=True))
    student.eval()

    # Load InverseAE (SOTA) model using the imported 3-channel U-Net architecture
    inverse_ae = InverseAEModel(in_channels=3, num_stains=2).to(device)
    inverse_ae.load_state_dict(torch.load(INVERSE_AE_WEIGHTS, map_location=device, weights_only=True))
    inverse_ae.eval()

    if not H5_PATH.exists():
        print(f"[!] Dataset not found at: {H5_PATH}. Aborting.")
        return

    with h5py.File(H5_PATH, "r") as f:
        N = f["rgb"].shape[0]
        labels = f["slide_label"][:] if "slide_label" in f else np.zeros(N)

        # Separate indices by cohort (Amyloid < 0.5, Tau >= 0.5)
        amyloid_candidates = np.where(labels < 0.5)[0]
        tau_candidates = np.where(labels >= 0.5)[0]

        # Select specific patches for visualization
        idx_amy = 7000 if 7000 < N and labels[7000] < 0.5 else amyloid_candidates[min(50, len(amyloid_candidates)-1)]
        idx_tau = 1000 if 1000 < N and labels[1000] >= 0.5 else tau_candidates[min(50, len(tau_candidates)-1)]

        sample_indices = [
            ("Specimen:\nAmyloid-β (6E10)", idx_amy),
            ("Specimen:\nTau (AT8)", idx_tau)
        ]

        fig, axes = plt.subplots(2, 7, figsize=(7.0, 2.3), dpi=300)
        plt.subplots_adjust(wspace=0.06, hspace=0.18, top=0.82, bottom=0.04, left=0.04, right=0.99)

        col_headers = [
            "Input\nRGB",
            "MorphoStain (Ours)\nRouting Mask $M_{\\mathrm{attn}}$",
            "MorphoStain (Ours)\nAmyloid-β (6E10)",
            "MorphoStain (Ours)\nTau (AT8)",
            "MorphoStain (Ours)\nHematoxylin",
            "InverseAE (SOTA)\nChannel 1",
            "InverseAE (SOTA)\nChannel 2"
        ]

        for row_idx, (cohort_name, p_idx) in enumerate(sample_indices):
            rgb_np = f["rgb"][p_idx]
            rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(device)

            with torch.no_grad():
                # --- MorphoStain Inference ---
                c_mean, _, attn_prob = student.forward_train(rgb_t)
                stains_3ch = student.get_positive_stains(rgb_t)

                s_abeta = stains_3ch[0, 0].cpu().numpy()
                s_tau = stains_3ch[0, 1].cpu().numpy()
                s_htx = stains_3ch[0, 2].cpu().numpy()

                c_dab_pure = s_abeta + s_tau
                
                # Accurately reflect the model's confidence without artificial smudging
                attn_smooth = np.clip(attn_prob[0, 0].cpu().numpy(), 0.0, 1.0)
                
                morpho_abeta = c_dab_pure * attn_smooth
                morpho_tau = c_dab_pure * (1.0 - attn_smooth)
                morpho_htx = s_htx

                # --- InverseAE Inference ---
                c_inv = inverse_ae(rgb_t)[0].cpu().numpy()
                invae_ch1 = c_inv[0]
                invae_ch2 = c_inv[1]

            orig_disp = np.clip(np.transpose(rgb_np, (1, 2, 0)), 0.0, 1.0)
            
            # Calibrate intensity limits based on the 99.5th percentile to handle outliers
            dab_peak = float(np.percentile(c_dab_pure, 99.5))
            htx_peak = float(np.percentile(morpho_htx, 99.5))

            vis_m_abeta = norm_calibrated(morpho_abeta, dab_peak)
            vis_m_tau = norm_calibrated(morpho_tau, dab_peak)
            vis_m_htx = norm_calibrated(morpho_htx, htx_peak)

            vis_inv_ch1 = norm_calibrated(invae_ch1, float(np.percentile(invae_ch1, 99.5)))
            vis_inv_ch2 = norm_calibrated(invae_ch2, float(np.percentile(invae_ch2, 99.5)))

            row_data = [
                (orig_disp, None),
                (attn_smooth, "coolwarm"),
                (vis_m_abeta, "viridis"),
                (vis_m_tau, "viridis"),
                (vis_m_htx, "viridis"),
                (vis_inv_ch1, "magma"),
                (vis_inv_ch2, "magma")
            ]

            for col_idx, (img, cmap) in enumerate(row_data):
                ax = axes[row_idx, col_idx]
                if cmap is None:
                    ax.imshow(img)
                else:
                    # Clip near-zero background noise in viridis/magma colormaps
                    ax.imshow(img, cmap=cmap, vmin=0.02, vmax=1.0)

                if row_idx == 0:
                    ax.set_title(col_headers[col_idx], fontsize=5.5, fontweight="bold", pad=3, linespacing=1.05)

                if col_idx == 0:
                    ax.set_ylabel(cohort_name, fontsize=6.0, fontweight="bold", labelpad=2)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis("off")

    save_path = RESULTS_DIR / "Disentanglement_Comparison.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[✓] Saved Calibrated Disentanglement Comparison Figure to: {save_path}")


if __name__ == "__main__":
    main()