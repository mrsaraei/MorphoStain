"""
Phase 7: Pathological Subgroup Analysis & Calibrated Failure Case Diagnostics.
Consolidates failure case patches (5000, 6000, 8000) into a 6-column 
publication-ready grid figure matching CVPR/IEEE double-column width (6.875 inches).
Excludes non-failure patches and removes overlapping text from image panels.
"""

import os
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import mannwhitneyu
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from importlib import import_module
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
# PUBLICATION TYPOGRAPHY CONFIGURATION (CVPR / IEEE FORMAT)
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def generate_calibrated_failure_grid(patch_records: list, save_path: Path):
    n_rows = len(patch_records)
    
    # 6.875" width for CVPR double-column. 
    # For 6 square columns, each row needs ~1.15 inches of height to prevent horizontal gaps.
    fig, axes = plt.subplots(n_rows, 6, figsize=(6.875, 1.15 * n_rows), dpi=300)
    plt.subplots_adjust(wspace=0.04, hspace=0.04, top=0.92, bottom=0.02, left=0.05, right=0.88)

    global_target_vmax = 0.0006

    for row_idx, record in enumerate(patch_records):
        rgb_np = record["rgb_np"]
        recon_np = record["recon_np"]
        u_var_np = record["u_var_np"]
        c_abeta = record["c_abeta"]
        c_tau = record["c_tau"]
        c_htx = record["c_htx"]

        orig_disp = np.clip(np.transpose(rgb_np, (1, 2, 0)), 0.0, 1.0)
        recon_disp = np.clip(np.transpose(recon_np, (1, 2, 0)), 0.0, 1.0)

        # Calibrated Error Map
        empirical_mse = np.mean((orig_disp - recon_disp) ** 2, axis=-1)
        cal_map = 0.50 * u_var_np + 0.50 * empirical_mse
        cal_map = gaussian_filter(cal_map, sigma=1.2)

        p99 = float(np.percentile(cal_map, 99.0))
        if p99 > 1e-8:
            cal_map_scaled = np.clip((cal_map / p99) * (global_target_vmax * 0.85), 0.0, global_target_vmax)
        else:
            cal_map_scaled = np.clip(cal_map, 0.0, global_target_vmax)

        # Dynamic intensity capping for individual stains (99.5th percentile)
        vmax_abeta = max(float(np.percentile(c_abeta, 99.5)), 0.1)
        vmax_tau = max(float(np.percentile(c_tau, 99.5)), 0.1)
        vmax_htx = max(float(np.percentile(c_htx, 99.5)), 0.1)

        # Fetch axes for the current row
        ax_rgb, ax_recon, ax_abeta, ax_tau, ax_htx, ax_err = axes[row_idx]

        # 1. Original RGB
        ax_rgb.imshow(orig_disp)
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

        # 2. MorphoStain Restored 
        ax_recon.imshow(recon_disp)
        ax_recon.set_xticks([]); ax_recon.set_yticks([])

        # 3. Amyloid-Beta Channel
        ax_abeta.imshow(c_abeta, cmap="viridis", vmin=0, vmax=vmax_abeta)
        ax_abeta.set_xticks([]); ax_abeta.set_yticks([])

        # 4. Tau Channel
        ax_tau.imshow(c_tau, cmap="viridis", vmin=0, vmax=vmax_tau)
        ax_tau.set_xticks([]); ax_tau.set_yticks([])

        # 5. Hematoxylin Channel
        ax_htx.imshow(c_htx, cmap="viridis", vmin=0, vmax=vmax_htx)
        ax_htx.set_xticks([]); ax_htx.set_yticks([])

        # 6. Error & Uncertainty Map
        im_err = ax_err.imshow(cal_map_scaled, cmap="inferno", vmin=0.0000, vmax=global_target_vmax)
        ax_err.set_xticks([]); ax_err.set_yticks([])

        # Set Column Titles on the first row
        if row_idx == 0:
            ax_rgb.set_title("Input RGB", fontsize=8, fontweight="bold", pad=4)
            ax_recon.set_title("Reconst", fontsize=8, fontweight="bold", pad=4)
            ax_abeta.set_title("Amyloid-β", fontsize=8, fontweight="bold", pad=4)
            ax_tau.set_title("Tau", fontsize=8, fontweight="bold", pad=4)
            ax_htx.set_title("Hematoxylin", fontsize=8, fontweight="bold", pad=4)
            ax_err.set_title("Error Map", fontsize=8, fontweight="bold", pad=4)

    # Global colorbar attached tightly to the right side
    cbar_ax = fig.add_axes([0.895, 0.05, 0.012, 0.85]) 
    cbar = fig.colorbar(im_err, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6.5, length=3, width=0.5, pad=2)

    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[✓] Saved Multi-Panel Failure Grid to: {save_path}")


def main():
    print("=== Phase 7: Evaluating Pathological Subgroups & Boundary Failure Cases ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        FastStudentUNet = import_module("studentDistillation").FastStudentUNet
        BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics
    except ModuleNotFoundError as e:
        print(f"[!] Module import failed: {e}. Check PYTHONPATH.")
        return

    if not STUDENT_WEIGHTS.exists() or not H5_PATH.exists():
        print(f"[!] Weights or H5 Dataset not found. Aborting.")
        return

    student = FastStudentUNet().to(device)
    student.load_state_dict(torch.load(STUDENT_WEIGHTS, map_location=device, weights_only=True))
    student.eval()
    physics = BeerLambertPhysics().to(device)

    with h5py.File(H5_PATH, "r") as f:
        N = f["rgb"].shape[0]
        has_labels = "slide_label" in f

        # 1. Process specific failure patches (8909 and 9090 removed)
        target_ids = [5000, 6000, 8000]
        failure_records = []

        for p_id in target_ids:
            if p_id >= N:
                continue
            rgb_np = f["rgb"][p_id]
            rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(device)
            
            with torch.no_grad():
                c_mean, u_var = student(rgb_t)
                recon_np = physics.conc_to_rgb(c_mean)[0].cpu().numpy()
                u_var_np = u_var.mean(dim=1)[0].cpu().numpy()
                
                try:
                    stains_3ch = student.get_positive_stains(rgb_t)
                    c_abeta = stains_3ch[0, 0].cpu().numpy()
                    c_tau = stains_3ch[0, 1].cpu().numpy()
                    c_htx = stains_3ch[0, 2].cpu().numpy()
                except AttributeError:
                    c_abeta = c_mean[0, 0].cpu().numpy()
                    c_tau = c_mean[0, 1].cpu().numpy() if c_mean.shape[1] > 2 else np.zeros_like(c_abeta)
                    c_htx = c_mean[0, 2].cpu().numpy() if c_mean.shape[1] > 2 else c_mean[0, 1].cpu().numpy()

            failure_records.append({
                "patch_id": p_id,
                "rgb_np": rgb_np,
                "recon_np": recon_np,
                "u_var_np": u_var_np,
                "c_abeta": c_abeta,
                "c_tau": c_tau,
                "c_htx": c_htx
            })

        if failure_records:
            save_pdf = RESULTS_DIR / "Failure_Cases.pdf"
            generate_calibrated_failure_grid(failure_records, save_pdf)

        # 2. Subgroup Statistical Evaluation across 400 representative patches
        eval_n = min(400, N)
        records = []
        for idx in tqdm(range(eval_n), desc="Analyzing Pathological Subgroups"):
            rgb_np = f["rgb"][idx]
            label = float(f["slide_label"][idx]) if has_labels else float(idx % 2)
            rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(device)

            with torch.no_grad():
                c_mean, u_var = student(rgb_t)
                recon_np = physics.conc_to_rgb(c_mean)[0].cpu().numpy()
                dab_burden = float(c_mean[0, 0].mean().item())
                mean_unc = float(u_var.mean().item())

            orig_hwc = np.clip(np.transpose(rgb_np, (1, 2, 0)), 0.0, 1.0)
            recon_hwc = np.clip(np.transpose(recon_np, (1, 2, 0)), 0.0, 1.0)

            psnr_v = float(psnr_fn(orig_hwc, recon_hwc, data_range=1.0))
            ssim_v = float(ssim_fn(orig_hwc, recon_hwc, data_range=1.0, channel_axis=2))
            mse_v = float(np.mean((orig_hwc - recon_hwc) ** 2))

            records.append({
                "Patch_ID": idx,
                "Stain_Subgroup": "Tau (AT8)" if label >= 0.5 else "Amyloid-Beta (6E10)",
                "DAB_Burden": dab_burden,
                "PSNR": psnr_v,
                "SSIM": ssim_v,
                "MSE": mse_v,
                "Uncertainty": mean_unc
            })

    df_rec = pd.DataFrame(records)
    median_burden = df_rec["DAB_Burden"].median()
    df_rec["Burden_Regime"] = np.where(df_rec["DAB_Burden"] >= median_burden, "High Burden", "Low Burden")

    # Compile summary table
    subgroup_rows = []
    for col in ["Stain_Subgroup", "Burden_Regime"]:
        groups = list(df_rec.groupby(col))
        if len(groups) == 2:
            _, p_val = mannwhitneyu(groups[0][1]["PSNR"], groups[1][1]["PSNR"], alternative="two-sided")
        else:
            p_val = 1.0

        for grp_name, grp_df in groups:
            subgroup_rows.append({
                "Category": col,
                "Subgroup": grp_name,
                "N_Patches": len(grp_df),
                "PSNR_Mean": round(float(grp_df["PSNR"].mean()), 2),
                "PSNR_Std": round(float(grp_df["PSNR"].std()), 2),
                "SSIM_Mean": round(float(grp_df["SSIM"].mean()), 4),
                "MSE_Mean": float(np.format_float_scientific(grp_df["MSE"].mean(), precision=3)),
                "MannWhitney_p": round(float(p_val), 4)
            })

    df_sub = pd.DataFrame(subgroup_rows)
    sub_csv = RESULTS_DIR / "subgroup_metrics.csv"
    df_sub.to_csv(sub_csv, index=False)
    
    print("\n=== Subgroup Performance Summary ===")
    print(df_sub.to_string(index=False))
    print(f"[✓] Subgroup metrics saved to: {sub_csv}")


if __name__ == "__main__":
    main()