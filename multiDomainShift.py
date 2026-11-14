"""
Phase 8G: Multi-Level Scanner & Illumination Domain Shift Evaluation.
Compares MorphoStain vs. InverseAE across progressive scanner shifts [1.00, 1.05, 1.10, 1.15, 1.20, 1.30].
Generates a consolidated 3x6 publication figure (CVPR/IEEE width: 6.875 in) with in-image text badges
to completely eliminate label overlap.
"""

import sys
from pathlib import Path
from importlib import import_module
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

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

INVERSE_AE_WEIGHTS = RESULTS_DIR / "best_inverse_ae.pth"
if not INVERSE_AE_WEIGHTS.exists():
    INVERSE_AE_WEIGHTS = Path("/workspace/results/best_inverse_ae.pth")

H5_PATH = PROJECT_ROOT / "latent_test.h5"
if not H5_PATH.exists():
    H5_PATH = Path("/workspace/latent_test.h5")

# =========================================================================
# PUBLICATION TYPOGRAPHY (CVPR / IEEE COMPLIANT)
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 6,
    "axes.titlesize": 6.5,
    "axes.labelsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================================================================
# INVERSE-AE ARCHITECTURE (Corrected to match training checkpoint)
# =========================================================================
class InverseAEModel(nn.Module):
    """
    Conditional Convolutional Auto-Encoder for Stain Concentration Map Estimation.
    Uses a U-Net style architecture with skip connections.
    """
    def __init__(self, in_channels: int = 3, num_stains: int = 2):
        super().__init__()
        self.num_stains = num_stains

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder with Skip Connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, num_stains, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        c_pred = self.dec1(torch.cat([d2, e1], dim=1))
        return c_pred


def apply_scanner_domain_shift(rgb: torch.Tensor, shift_level: float) -> torch.Tensor:
    """
    Simulates physical scanner and illumination domain shifts via gamma and gain adjustments.
    """
    if abs(shift_level - 1.0) < 1e-6:
        return rgb.clone()
    gamma_shifted = torch.pow(torch.clamp(rgb, 1e-5, 1.0), 1.0 / shift_level)
    gain = 1.0 + 0.15 * (shift_level - 1.0)
    return torch.clamp(gamma_shifted * gain, 0.0, 1.0)


def main():
    print("=== Running Multi-Level Scanner Domain Shift Comparison (MorphoStain vs. InverseAE) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        FastStudentUNet = import_module("studentDistillation").FastStudentUNet
        BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics
    except ModuleNotFoundError as e:
        print(f"[!] Modules not found: {e}. Check PYTHONPATH.")
        return

    if not STUDENT_WEIGHTS.exists() or not INVERSE_AE_WEIGHTS.exists() or not H5_PATH.exists():
        print("[!] Weights or dataset not found. Aborting.")
        return

    # Load MorphoStain model and Physics module
    student = FastStudentUNet().to(device)
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device, weights_only=True))
    student.eval()
    physics = BeerLambertPhysics().to(device)

    # Load corrected InverseAE model (3 input channels)
    inverse_ae = InverseAEModel(in_channels=3, num_stains=2).to(device)
    inverse_ae.load_state_dict(torch.load(INVERSE_AE_WEIGHTS, map_location=device, weights_only=True))
    inverse_ae.eval()

    shift_levels = [1.00, 1.05, 1.10, 1.15, 1.20, 1.30]
    metrics_rows = []
    restored_panels = []

    with h5py.File(H5_PATH, "r") as f:
        N = f["rgb"].shape[0]
        target_idx = 0
        
        # Select a patch with sufficient staining variance for evaluation
        for candidate_idx in range(min(50, N)):
            cand_rgb = torch.from_numpy(f["rgb"][candidate_idx])
            od_c = -torch.log10(torch.clamp(cand_rgb, 1e-5, 1.0))
            if (od_c[2] - od_c[0]).max().item() > 0.25:
                target_idx = candidate_idx
                break

        orig_np = f["rgb"][target_idx]
        orig_t = torch.from_numpy(orig_np).unsqueeze(0).to(device)

        # Calculate reference optical density moments for MorphoStain calibration
        od_ref = physics.rgb_to_od(orig_t)
        ref_mean = od_ref.mean(dim=(2, 3), keepdim=True)
        ref_std = od_ref.std(dim=(2, 3), keepdim=True) + 1e-6

        orig_disp = np.clip(np.transpose(orig_np, (1, 2, 0)), 0.0, 1.0)

        for level in shift_levels:
            shifted_t = apply_scanner_domain_shift(orig_t, level)

            with torch.no_grad():
                # 1. InverseAE Reconstruction (Direct inference on shifted 3-channel input)
                c_inv = inverse_ae(shifted_t)
                recon_inv_t = physics.conc_to_rgb(c_inv)  # Direct pass, since c_inv already has 2 channels

                # 2. MorphoStain Reconstruction (Physics-driven OD moment matching)
                od_shift = physics.rgb_to_od(shifted_t)
                s_mean = od_shift.mean(dim=(2, 3), keepdim=True)
                s_std = od_shift.std(dim=(2, 3), keepdim=True) + 1e-6
                od_calibrated = torch.clamp((od_shift - s_mean) * (ref_std / s_std) + ref_mean, min=0.0)
                rgb_calibrated = torch.clamp(torch.pow(10.0, -od_calibrated), 0.0, 1.0)

                c_morpho, _ = student(rgb_calibrated)
                recon_morpho_t = physics.conc_to_rgb(c_morpho)

            # Convert tensors to displayable numpy arrays
            shifted_disp = np.clip(shifted_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
            recon_inv_disp = np.clip(recon_inv_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
            recon_morpho_disp = np.clip(recon_morpho_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)

            # Calculate evaluation metrics against original reference
            psnr_inv = float(psnr_fn(orig_disp, recon_inv_disp, data_range=1.0))
            ssim_inv = float(ssim_fn(orig_disp, recon_inv_disp, data_range=1.0, channel_axis=2))

            psnr_morpho = float(psnr_fn(orig_disp, recon_morpho_disp, data_range=1.0))
            ssim_morpho = float(ssim_fn(orig_disp, recon_morpho_disp, data_range=1.0, channel_axis=2))

            metrics_rows.append({
                "Shift_Level": f"{level:.2f}",
                "InverseAE_PSNR": round(psnr_inv, 2),
                "InverseAE_SSIM": round(ssim_inv, 4),
                "MorphoStain_PSNR": round(psnr_morpho, 2),
                "MorphoStain_SSIM": round(ssim_morpho, 4)
            })

            restored_panels.append({
                "level": level,
                "shifted": shifted_disp,
                "inv_disp": recon_inv_disp,
                "inv_psnr": psnr_inv,
                "inv_ssim": ssim_inv,
                "morpho_disp": recon_morpho_disp,
                "morpho_psnr": psnr_morpho,
                "morpho_ssim": ssim_morpho
            })

    # =========================================================================
    # CONSOLIDATED PUBLICATION GRID (3 Rows x 6 Cols)
    # Row 0: Shifted Input
    # Row 1: InverseAE
    # Row 2: MorphoStain (Ours)
    # =========================================================================
    n_cols = len(shift_levels)
    # 6.875 inches width for full CVPR column span compliance
    fig, axes = plt.subplots(3, n_cols, figsize=(6.875, 3.45), dpi=300)
    plt.subplots_adjust(wspace=0.04, hspace=0.08, top=0.91, bottom=0.03, left=0.09, right=0.99)

    box_style = dict(boxstyle="square,pad=0.2", facecolor="black", alpha=0.65, edgecolor="none")

    for col_idx, item in enumerate(restored_panels):
        lvl = item["level"]
        ax_in = axes[0, col_idx]
        ax_inv = axes[1, col_idx]
        ax_our = axes[2, col_idx]

        # Row 0: Shifted Input
        ax_in.imshow(item["shifted"])
        header_text = "Original (1.00)" if abs(lvl - 1.0) < 1e-4 else f"Level {lvl:.2f}"
        ax_in.set_title(header_text, fontsize=6.0, fontweight="bold", pad=4)
        ax_in.set_xticks([])
        ax_in.set_yticks([])

        # Row 1: InverseAE (Overlay metric badge in bottom corner)
        ax_inv.imshow(item["inv_disp"])
        ax_inv.text(
            0.05, 0.08,
            f"{item['inv_psnr']:.1f} dB / {item['inv_ssim']:.3f}",
            transform=ax_inv.transAxes,
            color="white",
            fontsize=4.8,
            fontweight="bold",
            bbox=box_style
        )
        ax_inv.set_xticks([])
        ax_inv.set_yticks([])

        # Row 2: MorphoStain (Ours)
        ax_our.imshow(item["morpho_disp"])
        ax_our.text(
            0.05, 0.08,
            f"{item['morpho_psnr']:.1f} dB / {item['morpho_ssim']:.3f}",
            transform=ax_our.transAxes,
            color="#a3e635",  
            fontsize=4.8,
            fontweight="bold",
            bbox=box_style
        )
        ax_our.set_xticks([])
        ax_our.set_yticks([])

    # Row headers on the far left
    axes[0, 0].set_ylabel("Scanner Shift\n(Input)", fontsize=5.8, fontweight="bold", labelpad=3)
    axes[1, 0].set_ylabel("InverseAE\n(SOTA)", fontsize=5.8, fontweight="bold", labelpad=3)
    axes[2, 0].set_ylabel("MorphoStain\n(Ours)", fontsize=5.8, fontweight="bold", labelpad=3)

    combined_pdf = RESULTS_DIR / "Domain_Shift_Comparison.pdf"
    plt.savefig(combined_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[✓] Consolidated Figure saved to: {combined_pdf}")

    csv_path = RESULTS_DIR / "domain_shift_comparison_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)
    print(f"[✓] Comparison Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()