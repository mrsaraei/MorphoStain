"""
Phase 6A: Publication-Standard Figures Generator (Symmetric Morphological Routing).

This script dynamically generates publication-ready figures formatted for two-column
IEEE/CVPR paper layouts:
1. stain_unmixing_qualitative.pdf: Unified multi-row (N x 6) diagnostic figure showing all representative patches in one view.
2. PSNR_Distribution.pdf: Single-column KDE distribution of PSNR vs SOTA.
3. Convergence.pdf: Single-column training loss trajectory across ablation variants.
"""

import os
import sys
from pathlib import Path
from importlib import import_module
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import seaborn as sns
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

STUDENT_WEIGHTS = RESULTS_DIR / "best_student.pth"
if not STUDENT_WEIGHTS.exists():
    STUDENT_WEIGHTS = Path("/workspace/results/best_student.pth")

# =========================================================================
# PUBLICATION TYPOGRAPHY & MATPLOTLIB CONFIGURATION
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
    "pdf.fonttype": 42,  # TrueType fonts for CVPR/IEEE compliance
    "ps.fonttype": 42,
})


def norm_relative(arr: np.ndarray, ref_max: float) -> np.ndarray:
    """Scales channel proportionally to DAB peak to prevent stretching near-zero signals."""
    if ref_max < 1e-4:
        return np.zeros_like(arr)
    return np.clip(arr / ref_max, 0.0, 1.0)


def generate_patch_figures(student, physics, device: torch.device):
    """
    Renders all representative test patches in a single unified multi-row figure (Rows x 6 Columns).
    Width: 7.0 inches (Matches LaTeX double-column width).
    """
    if not H5_PATH.exists():
        print(f"[!] H5 dataset not found at: {H5_PATH}. Skipping patch figures.")
        return

    # Representative target indices spanning distinct pathological profiles
    target_indices = [10, 500, 2500, 6000, 10000]
    processed_patches = []

    with h5py.File(H5_PATH, "r") as f:
        N = f["rgb"].shape[0]
        for idx in target_indices:
            safe_idx = min(idx, N - 1)
            rgb_np = f["rgb"][safe_idx]
            rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(device)

            with torch.no_grad():
                c_mean, _, attn_prob = student.forward_train(rgb_t)
                recon_rgb = physics.conc_to_rgb(c_mean)[0].cpu().numpy()
                stains = student.get_positive_stains(rgb_t)

                s_abeta = stains[0, 0].cpu().numpy()
                s_tau = stains[0, 1].cpu().numpy()
                s_htx = stains[0, 2].cpu().numpy()

                c_dab_pure = s_abeta + s_tau

                attn_raw = attn_prob[0, 0].cpu().numpy()
                attn_smooth = np.clip(gaussian_filter(attn_raw, sigma=4.0), 0.0, 1.0)

                c_abeta_unbiased = c_dab_pure * attn_smooth
                c_tau_unbiased = c_dab_pure * (1.0 - attn_smooth)

                sum_abeta = float(np.sum(c_abeta_unbiased))
                sum_tau = float(np.sum(c_tau_unbiased))
                total_dab = sum_abeta + sum_tau + 1e-8

                pct_abeta = (sum_abeta / total_dab) * 100.0
                pct_tau = (sum_tau / total_dab) * 100.0

                dab_peak = float(np.percentile(c_dab_pure, 99.2))
                htx_peak = float(np.percentile(s_htx, 99.2))

                abeta_map = norm_relative(c_abeta_unbiased, dab_peak)
                tau_map = norm_relative(c_tau_unbiased, dab_peak)
                htx_map = norm_relative(s_htx, htx_peak)

            orig_img = np.clip(np.transpose(rgb_np, (1, 2, 0)), 0.0, 1.0)
            recon_img = np.clip(np.transpose(recon_rgb, (1, 2, 0)), 0.0, 1.0)

            processed_patches.append({
                "idx": safe_idx,
                "orig": orig_img,
                "recon": recon_img,
                "attn": attn_smooth,
                "abeta": abeta_map,
                "tau": tau_map,
                "htx": htx_map,
                "pct_abeta": pct_abeta,
                "pct_tau": pct_tau,
            })

    num_rows = len(processed_patches)
    if num_rows == 0:
        return

    # Height scales proportionally to the number of patches (~1.1 in per row)
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

    for row_idx, p in enumerate(processed_patches):
        row_items = [
            (p["orig"], None),
            (p["recon"], None),
            (p["attn"], "coolwarm"),
            (p["abeta"], "viridis"),
            (p["tau"], "viridis"),
            (p["htx"], "viridis"),
        ]

        for col_idx, (im, cmap) in enumerate(row_items):
            ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]

            if cmap is None:
                ax.imshow(im)
            else:
                ax.imshow(im, cmap=cmap, vmin=0.0, vmax=1.0)

            # Titles on top row only to eliminate vertical clutter
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=6.0, fontweight="bold", pad=3, linespacing=1.05)

            # Left row labels with patch index and quantitative stain split
            if col_idx == 0:
                row_label = f"Patch #{p['idx']}\n(Aβ:{p['pct_abeta']:.0f}% / Tau:{p['pct_tau']:.0f}%)"
                ax.set_ylabel(row_label, fontsize=5.5, fontweight="bold", labelpad=3)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")

    out_file = RESULTS_DIR / "stain_unmixing_qualitative.pdf"
    plt.savefig(out_file, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[✓] Saved unified multi-patch figure to: {out_file}")


def generate_psnr_distribution():
    """Plots kernel density estimation comparing MorphoStain vs SOTA."""
    dist_csv = RESULTS_DIR / "psnr_distributions.csv"
    if not dist_csv.exists():
        print(f"[!] File not found: {dist_csv}")
        return
    df = pd.read_csv(dist_csv)

    plt.figure(figsize=(3.3, 2.3), dpi=300)

    baseline_col = "InverseAE_PSNR" if "InverseAE_PSNR" in df.columns else "NMF_PSNR"
    baseline_label = "InverseAE (SOTA)" if baseline_col == "InverseAE_PSNR" else "Baseline (NMF)"
    baseline_color = "#17becf" if baseline_col == "InverseAE_PSNR" else "#f5a623"

    sns.kdeplot(
        data=df,
        x=baseline_col,
        fill=True,
        color=baseline_color,
        alpha=0.35,
        linewidth=1.2,
        label=baseline_label,
    )

    sns.kdeplot(
        data=df,
        x="Student_PSNR",
        fill=True,
        color="#1f77b4",
        alpha=0.45,
        linewidth=1.3,
        label="MorphoStain (Ours)",
    )

    plt.xlabel("PSNR (dB)", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.legend(frameon=False, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout(pad=0.2)

    out_path = RESULTS_DIR / "PSNR_Distribution.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[✓] Saved Synchronized PSNR Distribution Plot to: {out_path}")


def generate_convergence_plot():
    """Plots multi-variant loss trajectories across teacher ablation training."""
    variants = [
        ("loss_M1_MorphoStain.csv", "M1: MorphoStain", "#1f77b4", "-"),
        ("loss_M2_NoPhikon.csv", "M2: No Phikon (CNN only)", "#ff7f0e", "--"),
        ("loss_M3_NoPhysics.csv", "M3: No Physics Loss", "#2ca02c", "-."),
        ("loss_M4_NoOverlap.csv", "M4: No Overlap Penalty", "#d62728", ":"),
    ]

    plt.figure(figsize=(3.3, 2.3), dpi=300)
    plotted = False

    for fname, label, color, ls in variants:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            plt.plot(
                df["epoch"],
                df["loss"],
                linestyle=ls,
                color=color,
                marker="o",
                markersize=2.0,
                linewidth=1.1,
                label=label,
            )
            plotted = True

    if plotted:
        plt.xlabel("Epochs", fontweight="bold")
        plt.ylabel("Training Loss", fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.ylim(0.95, 1.45)
        plt.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.85,
            loc="upper right",
            fontsize=6.5,
        )

        plt.tight_layout(pad=0.2)
        out_path = RESULTS_DIR / "Convergence.pdf"
        plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        print(f"[✓] Saved Convergence Plot to: {out_path}")
    plt.close()


def main():
    print("=== Phase 6A: Generating Unified Publication Figures ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        FastStudentUNet = import_module("studentDistillation").FastStudentUNet
        BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics
    except ModuleNotFoundError:
        import sys
        code_dir = PROJECT_ROOT / "code"
        if code_dir.exists():
            sys.path.append(str(code_dir))
        FastStudentUNet = import_module("studentDistillation").FastStudentUNet
        BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics

    if STUDENT_WEIGHTS.exists():
        student = FastStudentUNet().to(device)
        student.load_state_dict(
            torch.load(STUDENT_WEIGHTS, map_location=device, weights_only=True)
        )
        student.eval()
        physics = BeerLambertPhysics().to(device)
        generate_patch_figures(student, physics, device)
    else:
        print(f"[!] Model weights not found at: {STUDENT_WEIGHTS}. Skipping patch figure.")

    generate_psnr_distribution()
    generate_convergence_plot()


if __name__ == "__main__":
    main()