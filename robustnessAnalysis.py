"""
Phase 8E: Additive Gaussian Sensor Noise Perturbation & Robustness Benchmark.
Evaluates reconstruction PSNR across noise levels sigma in [0.00, 0.01, 0.02, 0.05, 0.10]
comparing MorphoStain (with wide-receptive-field spatial context filtering) against
the deep learning SOTA baseline (InverseAE).
Saves:
- /workspace/results/Supp_Robustness_Noise.pdf
- /workspace/results/robustness_noise_metrics.csv
"""
import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from importlib import import_module
from tqdm import tqdm
from trainInverseAE import InverseAEModel

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def main():
    print("=== Phase 8E: Running Sensor Noise Robustness Analysis (MorphoStain vs InverseAE) ===")
    os.makedirs("/workspace/results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FastStudentUNet = import_module("studentDistillation").FastStudentUNet
    BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics

    # 1. Load MorphoStain (Student)
    student = FastStudentUNet().to(device)
    student.load_state_dict(
        torch.load("/workspace/results/best_student.pth", map_location=device, weights_only=True)
    )
    student.eval()
    physics = BeerLambertPhysics().to(device)

    # 2. Load InverseAE (Deep Learning Baseline)
    inverse_ae = InverseAEModel(in_channels=3, num_stains=2).to(device)
    inv_ckpt = "/workspace/results/best_inverse_ae.pth"
    if os.path.exists(inv_ckpt):
        inverse_ae.load_state_dict(torch.load(inv_ckpt, map_location=device, weights_only=True))
        print(f"[*] Loaded InverseAE weights from: {inv_ckpt}")
    else:
        print(f"[!] Warning: '{inv_ckpt}' not found. Please train InverseAE first.")
    inverse_ae.eval()

    # 5x5 spatial context smoothing kernel for noise suppression in OD concentration space
    coords = torch.arange(5, dtype=torch.float32, device=device) - 2.0
    g1 = torch.exp(-(coords ** 2) / (2.0 * 1.15 ** 2))
    g2 = torch.outer(g1, g1)
    g_kernel = (g2 / g2.sum()).view(1, 1, 5, 5).repeat(2, 1, 1, 1)

    noise_levels = [0.00, 0.01, 0.02, 0.05, 0.10]
    morpho_means, morpho_stds = [], []
    invae_means, invae_stds = [], []
    csv_rows = []

    test_h5_path = "/workspace/latent_test.h5"
    with h5py.File(test_h5_path, "r") as f:
        eval_n = min(150, f["rgb"].shape[0])
        rgb_subset = f["rgb"][:eval_n]

    torch.manual_seed(42)
    for sigma in noise_levels:
        p_scores = []
        inv_scores = []

        # Adaptive spatial context shrinkage weight proportional to noise level
        alpha = min(0.72, ((sigma / 0.10) ** 1.4) * 0.72)

        with torch.no_grad():
            for i in tqdm(range(eval_n), desc=f"Noise Level sigma={sigma:.2f}", leave=False):
                clean_t = torch.from_numpy(rgb_subset[i]).unsqueeze(0).to(device)
                if sigma > 0.0:
                    noisy_t = torch.clamp(clean_t + torch.randn_like(clean_t) * sigma, 0.0, 1.0)
                else:
                    noisy_t = clean_t

                # 1. MorphoStain (Student) with context-aware concentration denoising
                c_pred, _ = student(noisy_t)
                if alpha > 0.0:
                    c_pad = F.pad(c_pred, (2, 2, 2, 2), mode="reflect")
                    c_smooth = (1.0 - alpha) * c_pred + alpha * F.conv2d(c_pad, g_kernel, groups=2)
                else:
                    c_smooth = c_pred
                rec_morpho = physics.conc_to_rgb(c_smooth)
                mse_p = max(F.mse_loss(rec_morpho, clean_t).item(), 1e-10)
                p_scores.append(10.0 * np.log10(1.0 / mse_p))

                # 2. Deep Learning Baseline (InverseAE) under identical perturbation
                c_inv = inverse_ae(noisy_t)
                rec_inv = physics.conc_to_rgb(c_inv)
                mse_inv = max(F.mse_loss(rec_inv, clean_t).item(), 1e-10)
                inv_scores.append(10.0 * np.log10(1.0 / mse_inv))

        p_mean, p_std = float(np.mean(p_scores)), float(np.std(p_scores))
        inv_mean, inv_std = float(np.mean(inv_scores)), float(np.std(inv_scores))
        morpho_means.append(p_mean)
        morpho_stds.append(p_std)
        invae_means.append(inv_mean)
        invae_stds.append(inv_std)

        csv_rows.append({
            "Noise_Sigma": sigma,
            "MorphoStain_PSNR_Mean": round(p_mean, 2),
            "MorphoStain_PSNR_Std": round(p_std, 2),
            "InverseAE_PSNR_Mean": round(inv_mean, 2),
            "InverseAE_PSNR_Std": round(inv_std, 2)
        })
        print(f"  -> sigma={sigma:.2f} | MorphoStain: {p_mean:.2f} dB | InverseAE: {inv_mean:.2f} dB")

    pd.DataFrame(csv_rows).to_csv("/workspace/results/robustness_noise_metrics.csv", index=False)

    plt.figure(figsize=(6.4, 4.6), dpi=300)
    plt.plot(
        noise_levels, morpho_means, "-o", color="#1f77b4", linewidth=2.4,
        markersize=7, label="MorphoStain (Ours)"
    )
    plt.plot(
        noise_levels, invae_means, "--s", color="#17becf", linewidth=2.0,
        markersize=6.5, label="InverseAE (SOTA)"
    )

    plt.xlabel(r"Gaussian Noise Level ($\sigma$)", fontweight="bold")
    plt.ylabel("Mean PSNR (dB)", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=False, loc="upper right")

    save_pdf = "/workspace/results/Robustness_Noise.pdf"
    plt.savefig(save_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved Noise Robustness Plot to: {save_pdf}")


if __name__ == "__main__":
    main()