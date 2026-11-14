"""
Phase 9: Uncertainty Calibration & Reliability Diagram Generator.
Aligns quantile-binned predicted uncertainty U(x) with empirical reconstruction MSE
and outputs publication-grade single-column vector graphics (CVPR/IEEE compliant).
"""

import os
import sys
from pathlib import Path
from importlib import import_module
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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

H5_PATH = PROJECT_ROOT / "latent_test.h5"
if not H5_PATH.exists():
    H5_PATH = Path("/workspace/latent_test.h5")

# =========================================================================
# CVPR / IEEE SINGLE-COLUMN TYPOGRAPHY CONFIGURATION
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # Mandatory Type 42 TrueType font embedding
    "ps.fonttype": 42,
})


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Calculating Calibrated Reliability Curve on: {device.type.upper()}")

    try:
        FastStudentUNet = import_module("studentDistillation").FastStudentUNet
        BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics
    except ModuleNotFoundError as e:
        print(f"[!] Import failed: {e}. Check PYTHONPATH.")
        return

    if not STUDENT_WEIGHTS.exists():
        print(f"[!] Checkpoint not found at: {STUDENT_WEIGHTS}. Aborting.")
        return

    student = FastStudentUNet().to(device)
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device, weights_only=True))
    student.eval()
    physics = BeerLambertPhysics().to(device)

    if not H5_PATH.exists():
        print(f"[!] Evaluation file missing at: {H5_PATH}. Aborting.")
        return

    with h5py.File(H5_PATH, "r") as f:
        eval_subset_size = min(300, len(f["rgb"]))
        rgb_data = f["rgb"][:eval_subset_size]

    predicted_uncertainties = []
    actual_errors = []

    np.random.seed(42)

    with torch.no_grad():
        for i in tqdm(range(eval_subset_size), desc="Evaluating Uncertainty Calibration"):
            rgb_tensor = torch.from_numpy(rgb_data[i]).unsqueeze(0).to(device)
            c_mean, u_var = student(rgb_tensor)
            recon_tensor = physics.conc_to_rgb(c_mean)

            error_map = torch.mean((recon_tensor - rgb_tensor) ** 2, dim=1).squeeze().cpu().numpy()
            uncertainty_map = u_var.squeeze().mean(dim=0).cpu().numpy()

            # Representative pixel sampling
            sub_idx = np.random.choice(error_map.size, size=512, replace=False)
            predicted_uncertainties.append(uncertainty_map.flatten()[sub_idx])
            actual_errors.append(error_map.flatten()[sub_idx])

    predicted_uncertainties = np.concatenate(predicted_uncertainties)
    actual_errors = np.concatenate(actual_errors)

    # Robust percentile-based normalization onto diagnostic range [0.0, 1.0]
    u_max = np.percentile(predicted_uncertainties, 99.5) + 1e-8
    e_max = np.percentile(actual_errors, 99.5) + 1e-8

    u_norm = np.clip(predicted_uncertainties / u_max, 0.0, 1.0)
    e_norm = np.clip(actual_errors / e_max, 0.0, 1.0)

    # Binning by quantile for empirical reliability diagram
    num_bins = 10
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_uncertainty_means = []
    bin_error_means = []
    bin_error_stds = []
    bin_weights = []

    for i in range(num_bins):
        mask = (u_norm >= bin_edges[i]) & (u_norm < bin_edges[i + 1]) if i < num_bins - 1 else (u_norm >= bin_edges[i]) & (u_norm <= bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_uncertainty_means.append(np.mean(u_norm[mask]))
            bin_error_means.append(np.mean(e_norm[mask]))
            bin_error_stds.append(np.std(e_norm[mask]) / np.sqrt(np.sum(mask)))  # Standard error
            bin_weights.append(np.sum(mask) / len(u_norm))
        else:
            bin_uncertainty_means.append(bin_centers[i])
            bin_error_means.append(bin_centers[i])
            bin_error_stds.append(0.0)
            bin_weights.append(0.0)

    bin_uncertainty_means = np.array(bin_uncertainty_means)
    bin_error_means = np.array(bin_error_means)
    bin_error_stds = np.array(bin_error_stds)
    bin_weights = np.array(bin_weights)

    # Expected Calibration Error (ECE)
    ece = float(np.sum(bin_weights * np.abs(bin_uncertainty_means - bin_error_means)))

    # Exact Single-Column Width for IEEE/CVPR (3.25 in x 2.6 in)
    fig, ax = plt.subplots(figsize=(3.25, 2.6), dpi=300)

    # Ideal diagonal
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#555555", linewidth=1.1, label="Ideal Calibration")

    # Empirical calibration line with standard error shading
    ax.plot(
        bin_uncertainty_means,
        bin_error_means,
        marker="o",
        linestyle="-",
        color="#08519c",
        linewidth=1.4,
        markersize=4.0,
        label=f"MorphoStain (ECE = {ece:.3f})"
    )
    ax.fill_between(
        bin_uncertainty_means,
        np.clip(bin_error_means - 1.96 * bin_error_stds, 0.0, 1.0),
        np.clip(bin_error_means + 1.96 * bin_error_stds, 0.0, 1.0),
        color="#3182bd",
        alpha=0.22,
        label=r"95% CI"
    )

    ax.set_xlabel(r"Normalized Predicted Uncertainty $\hat{\mathcal{U}}(x)$", fontweight="bold")
    ax.set_ylabel(r"Empirical Reconstruction Error $\hat{\mathcal{E}}(x)$", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5, color="#b0b0b0")
    ax.legend(frameon=False, loc="upper left", handlelength=1.5)

    save_path = RESULTS_DIR / "Reliability_Diagram.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"[✓] Saved publication-compliant Reliability Diagram to: {save_path}")
    print(f"[*] Computed Expected Calibration Error (ECE): {ece:.4f}")


if __name__ == "__main__":
    main()