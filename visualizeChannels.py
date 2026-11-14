"""
Phase 6B: Morphological Routing & Stain Disentanglement Channel Visualizer.
Generates a unified publication-ready multi-row figure (5x6 Grid) matching exact
CVPR/IEEE double-column width (7.0 in) with vector fonts (Type 42) and compact vertical layout:
  [Original RGB, MorphoStain Recon, Routing Mask M_attn, Amyloid-Beta, Tau (AT8), Hematoxylin]
"""

import sys
from pathlib import Path
from importlib import import_module
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

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

H5_PATH = PROJECT_ROOT / "latent_test.h5"
if not H5_PATH.exists():
    H5_PATH = Path("/workspace/latent_test.h5")

WEIGHTS_PATH = RESULTS_DIR / "best_student.pth"
if not WEIGHTS_PATH.exists():
    WEIGHTS_PATH = Path("/workspace/results/best_student.pth")

# =========================================================================
# PUBLICATION TYPOGRAPHY & MATPLOTLIB CONFIGURATION
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  
    "ps.fonttype": 42,
})


def norm_relative(arr: np.ndarray, ref_max: float) -> np.ndarray:
    """Scales channel proportionally to DAB peak to prevent stretching near-zero signals."""
    if ref_max < 1e-4:
        return np.zeros_like(arr)
    return np.clip(arr / ref_max, 0.0, 1.0)


def main():
    print("=== Generating Unified Multi-Patch Morphological Routing Visualizer ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        FastStudentUNet = import_module("studentDistillation").FastStudentUNet
        BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics
    except ModuleNotFoundError:
        print("[!] Error loading network modules. Ensure PYTHONPATH includes code/ directory.")
        return

    if not WEIGHTS_PATH.exists():
        print(f"[!] Model weights not found at: {WEIGHTS_PATH}. Aborting.")
        return

    student = FastStudentUNet().to(device)
    student.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    )
    student.eval()
    physics = BeerLambertPhysics().to(device)

    if not H5_PATH.exists():
        print(f"[!] H5 dataset not found at: {H5_PATH}. Cannot render patches.")
        return

    candidate_indices = [32, 550, 2830, 9904, 6600]
    extracted_records = []

    with h5py.File(H5_PATH, "r") as f:
        N = f["rgb"].shape[0]

        for p_idx in candidate_indices:
            safe_idx = min(p_idx, N - 1)
            rgb_np = f["rgb"][safe_idx]
            rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(device)

            with torch.no_grad():
                c_mean, _, attn_prob = student.forward_train(rgb_t)
                recon_rgb = physics.conc_to_rgb(c_mean)[0].cpu().numpy()
                stains_3ch = student.get_positive_stains(rgb_t)

                s_abeta = stains_3ch[0, 0].cpu().numpy()
                s_tau = stains_3ch[0, 1].cpu().numpy()
                s_htx = stains_3ch[0, 2].cpu().numpy()

                c_dab_pure = s_abeta + s_tau

                attn_np = attn_prob[0, 0].cpu().numpy()
                attn_smooth = gaussian_filter(attn_np, sigma=4.0)
                attn_smooth = np.clip(attn_smooth, 0.0, 1.0)

                # Pure symmetric morphological routing
                c_abeta_unbiased = c_dab_pure * attn_smooth
                c_tau_unbiased = c_dab_pure * (1.0 - attn_smooth)

                sum_abeta = float(np.sum(c_abeta_unbiased))
                sum_tau = float(np.sum(c_tau_unbiased))
                total_dab = sum_abeta + sum_tau + 1e-8

                pct_abeta = (sum_abeta / total_dab) * 100.0
                pct_tau = (sum_tau / total_dab) * 100.0

                dab_peak = float(np.percentile(c_dab_pure, 99.2))
                htx_peak = float(np.percentile(s_htx, 99.2))

                abeta_vis = norm_relative(c_abeta_unbiased, dab_peak)
                tau_vis = norm_relative(c_tau_unbiased, dab_peak)
                htx_vis = norm_relative(s_htx, htx_peak)

            orig_disp = np.clip(np.transpose(rgb_np, (1, 2, 0)), 0.0, 1.0)
            recon_disp = np.clip(np.transpose(recon_rgb, (1, 2, 0)), 0.0, 1.0)

            extracted_records.append({
                "patch_id": safe_idx,
                "orig": orig_disp,
                "recon": recon_disp,
                "attn": attn_smooth,
                "abeta": abeta_vis,
                "tau": tau_vis,
                "htx": htx_vis,
                "pct_abeta": pct_abeta,
                "pct_tau": pct_tau,
            })

    num_rows = len(extracted_records)
    if num_rows == 0:
        return

    # Proportional figure dimensions for LaTeX 2-column format (width: 7.0 in)
    fig_height = 0.35 + num_rows * 1.15
    fig, axes = plt.subplots(num_rows, 6, figsize=(7.0, fig_height), dpi=300)
    plt.subplots_adjust(wspace=0.04, hspace=0.10, top=0.92, bottom=0.03, left=0.08, right=0.99)

    col_headers = [
        "Input\nRGB",
        "MorphoStain\nRecon",
        "Routing Mask\n$M_{\\mathrm{attn}}$",
        "Amyloid-β\n(6E10)",
        "Tau\n(AT8)",
        "Hematoxylin"
    ]

    for row_idx, rec in enumerate(extracted_records):
        row_items = [
            (rec["orig"], None),
            (rec["recon"], None),
            (rec["attn"], "coolwarm"),
            (rec["abeta"], "viridis"),
            (rec["tau"], "viridis"),
            (rec["htx"], "viridis"),
        ]

        for col_idx, (img_data, cmap) in enumerate(row_items):
            ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]

            if cmap is None:
                ax.imshow(img_data)
            else:
                ax.imshow(img_data, cmap=cmap, vmin=0.0, vmax=1.0)

            # Column header titles placed strictly on row 0
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=6.0, fontweight="bold", pad=3, linespacing=1.05)

            # Patch info and quantitative split metrics along the left axis
            if col_idx == 0:
                ax.set_ylabel(
                    f"Patch #{rec['patch_id']}\n(Aβ:{rec['pct_abeta']:.0f}% / Tau:{rec['pct_tau']:.0f}%)",
                    fontsize=5.5,
                    fontweight="bold",
                    labelpad=3
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")

    unified_output = RESULTS_DIR / "Morphological_Routing.pdf"
    plt.savefig(unified_output, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[✓] Saved unified routing comparison figure to: {unified_output}")


if __name__ == "__main__":
    main()