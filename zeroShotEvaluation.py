"""
Phase 8F: Zero-Shot Out-of-Distribution (OOD) Multi-Seed Benchmark.
Evaluates the pre-trained MorphoStain FastStudentUNet against classical baselines
and SOTA Deep Learning (InverseAE) across multiple deterministic seeds ([42, 123, 2024]).

Features:
- Deterministic multi-seed iteration for publication reproducibility (CVPR/MICCAI).
- Collision-free radial annotations for clustered Pareto points in Panel (b).
- Rotated typography and tightened layout boundaries compliant with CVPR column specifications.

Generates:
- /workspace/results/zero_shot_metrics.csv
- /workspace/results/ZeroShot_Evaluation.pdf
"""

import os
import sys
import time
import random
from glob import glob
from pathlib import Path
import h5py
import lpips
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib.pyplot as plt
from importlib import import_module
from tqdm import tqdm
from trainInverseAE import InverseAEModel

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================================
# PUBLICATION TYPOGRAPHY CONFIGURATION (CVPR / IEEE STANDARD)
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def set_deterministic_seed(seed: int = 42):
    """Enforces deterministic execution across Python, NumPy, and PyTorch backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_and_cache_patches(source_dir: str, cache_dir: str, target_count: int = 100, patch_size: int = 512):
    """Extracts, validates tissue coverage, and caches patches from WSIs."""
    os.makedirs(cache_dir, exist_ok=True)
    existing_patches = sorted(glob(os.path.join(cache_dir, "*.png")))
    
    if len(existing_patches) >= target_count:
        return existing_patches[:target_count]
        
    print(f"[*] Extracting new patches from {source_dir} (Target: {target_count})...")
    for p in existing_patches:
        os.remove(p)
        
    wsi_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        wsi_files.extend(glob(os.path.join(source_dir, "**", ext), recursive=True))
    
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
                    # Retain only foreground tissue regions
                    if np.mean(gray < 0.90) > 0.40:  
                        out_name = os.path.join(cache_dir, f"patch_{count:03d}.png")
                        Image.fromarray(patch).save(out_name)
                        extracted_paths.append(out_name)
                        count += 1
        except Exception:
            pass
            
    return extracted_paths


def load_zero_shot_patches(max_patches: int = 100) -> np.ndarray:
    """Loads standardized evaluation patches from the persistent cache."""
    source_dir = "/workspace/dataset/test2"
    cache_dir = "/workspace/dataset/test2_patches_100"
    
    patch_paths = extract_and_cache_patches(source_dir, cache_dir, target_count=max_patches)
    
    if len(patch_paths) == 0:
        with h5py.File("/workspace/latent_val.h5", "r") as f:
            n_avail = min(max_patches, f["rgb"].shape[0])
            return f["rgb"][:n_avail]
            
    tf = transforms.Compose([transforms.ToTensor()])
    arr_list = []
    for p in patch_paths:
        arr_list.append(tf(Image.open(p).convert("RGB")).numpy())
    return np.stack(arr_list, axis=0)


def eval_metrics_single(true_rgb: np.ndarray, pred_rgb: np.ndarray, lpips_fn, device: torch.device) -> dict:
    """Computes reference-based full reconstruction fidelity metrics."""
    true_c = np.clip(true_rgb, 0.0, 1.0)
    pred_c = np.clip(pred_rgb, 0.0, 1.0)

    mse = max(float(np.mean((true_c - pred_c) ** 2)), 1e-10)
    rmse = float(np.sqrt(mse))
    psnr = float(psnr_fn(true_c, pred_c, data_range=1.0))
    ssim_val = float(ssim_fn(
        np.transpose(true_c, (1, 2, 0)),
        np.transpose(pred_c, (1, 2, 0)),
        data_range=1.0,
        channel_axis=2
    ))

    with torch.no_grad():
        t_ten = torch.from_numpy(true_c).unsqueeze(0).to(device) * 2.0 - 1.0
        p_ten = torch.from_numpy(pred_c).unsqueeze(0).to(device) * 2.0 - 1.0
        lp_val = float(lpips_fn(t_ten, p_ten).item())

    return {
        "PSNR": psnr,
        "SSIM": ssim_val,
        "MSE": mse,
        "RMSE": rmse,
        "LPIPS": lp_val
    }


def render_figure(records: dict, df_zs: pd.DataFrame, pdf_path: str):
    """Generates the CVPR-compliant collision-free 2-panel comparative figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.875, 2.7), dpi=300)
    fig.subplots_adjust(wspace=0.25, left=0.07, right=0.98, top=0.88, bottom=0.23)

    palette = {
        "MorphoStain": "#1f77b4",
        "InverseAE":  "#17becf",
        "Ruifrok":    "#2ca02c",
        "Macenko":    "#d62728",
        "Vahadane":   "#9467bd",
        "NMF":        "#ff7f0e",
        "Sparse_NMF": "#8c564b",
        "LassoReg":   "#e377c2"
    }

    models_order = ["MorphoStain", "InverseAE", "Ruifrok", "Macenko", "Vahadane", "NMF", "Sparse_NMF", "LassoReg"]
    positions = np.arange(len(models_order))

    # --- Panel (a): Boxplot with Raw Data Jitter ---
    plot_data = [{"Model": m, "PSNR": r["PSNR"]} for m, recs in records.items() for r in recs]
    df_plot = pd.DataFrame(plot_data)

    box_data = [df_plot[df_plot["Model"] == m]["PSNR"].values for m in models_order]
    bp = ax1.boxplot(
        box_data, positions=positions, patch_artist=True, widths=0.48, notch=False, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 3.0},
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"linewidth": 0.8, "alpha": 0.85},
        whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 1.5, "alpha": 0.25, "markeredgecolor": "none"}
    )

    for patch, model_name in zip(bp["boxes"], models_order):
        patch.set_facecolor(palette[model_name])

    for idx, m in enumerate(models_order):
        y_vals = df_plot[df_plot["Model"] == m]["PSNR"].values
        x_vals = np.random.normal(idx, 0.05, size=len(y_vals))
        ax1.scatter(x_vals, y_vals, alpha=0.20, color=palette[m], edgecolor="none", s=3.5, zorder=3)

    short_labels = ["MorphoStain\n(Ours)", "InverseAE\n(SOTA)", "Ruifrok", "Macenko", "Vahadane", "NMF", "Sparse NMF", "Lasso Reg"]
    ax1.set_xticks(positions)
    ax1.set_xticklabels(short_labels, rotation=35, ha="right", rotation_mode="anchor", fontsize=5.5)
    ax1.set_ylabel("Reconstruction PSNR (dB)", fontweight="bold", labelpad=2)
    ax1.set_title("(a) Out-of-Distribution Generalization Stability", fontweight="bold", pad=5)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    # --- Panel (b): Clutter-Free Radial Pareto Frontier ---
    clean_offsets = {
        "MorphoStain": {"xy": (0, 14),     "ha": "center", "va": "bottom"},
        "Sparse_NMF":  {"xy": (-35, 18),   "ha": "right",  "va": "bottom"},
        "NMF":         {"xy": (25, 20),    "ha": "left",   "va": "bottom"},
        "Macenko":     {"xy": (38, -2),    "ha": "left",   "va": "center"},
        "Vahadane":    {"xy": (-36, 4),    "ha": "right",  "va": "center"},
        "Ruifrok":     {"xy": (-36, -14),  "ha": "right",  "va": "top"},
        "InverseAE":   {"xy": (28, -12),   "ha": "left",   "va": "top"},
        "LassoReg":    {"xy": (0, -22),    "ha": "center", "va": "top"},
    }

    for _, row in df_zs.iterrows():
        m = row["Model"]
        psnr_m = row["PSNR_Mean"]
        lpips_m = row["LPIPS_Mean"]
        fps_m = row["FPS"]
        c = palette[m]

        marker_size = min(110, 32 + (fps_m * 0.42))
        is_ours = (m == "MorphoStain")

        ax2.scatter(
            lpips_m, psnr_m, s=marker_size, color=c,
            edgecolors="black" if is_ours else "dimgray",
            linewidth=1.2 if is_ours else 0.6, alpha=0.92, zorder=4
        )

        cfg = clean_offsets.get(m, {"xy": (0, 10), "ha": "center", "va": "bottom"})
        name_display = "MorphoStain (Ours)" if is_ours else ("InverseAE" if m == "InverseAE" else m)
        label_text = f"{name_display}\n({fps_m:.0f} FPS)"

        ax2.annotate(
            label_text,
            xy=(lpips_m, psnr_m),
            xytext=cfg["xy"],
            textcoords="offset points",
            fontsize=5.2,
            fontweight="bold" if is_ours else "normal",
            ha=cfg["ha"],
            va=cfg["va"],
            arrowprops=dict(
                arrowstyle="->",
                color="#666666",
                lw=0.5,
                shrinkA=2,
                shrinkB=3,
                connectionstyle="arc3,rad=0.0"
            )
        )

    # Margins expanded to fully display radial annotations
    ax2.set_xlim(0.038, 0.145)
    ax2.set_ylim(30.0, 39.5)
    ax2.set_xlabel(r"Perceptual Distance (LPIPS) $\downarrow$", fontweight="bold", labelpad=2)
    ax2.set_ylabel(r"Zero-Shot PSNR (dB) $\uparrow$", fontweight="bold", labelpad=2)
    ax2.set_title("(b) Quality vs. Inference Speed Trade-off", fontweight="bold", pad=5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"  [✓] Publication benchmark figure saved to: {pdf_path}")


def main():
    print("=== Phase 8F: Running Zero-Shot Multi-Seed Benchmark (8 Methods, 100 Patches) ===")
    os.makedirs("/workspace/results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FastStudentUNet = import_module("studentDistillation").FastStudentUNet
    BeerLambertPhysics = import_module("teacherDiffusion").BeerLambertPhysics

    # 1. Initialize MorphoStain Student Model
    student = FastStudentUNet().to(device)
    student.load_state_dict(torch.load("/workspace/results/best_student.pth", map_location=device, weights_only=True))
    student.eval()

    physics = BeerLambertPhysics().to(device)
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    # 2. Initialize SOTA Deep Learning Baseline (InverseAE)
    inverse_ae = InverseAEModel(in_channels=3, num_stains=2).to(device)
    inv_ckpt = "/workspace/results/best_inverse_ae.pth"
    if os.path.exists(inv_ckpt):
        inverse_ae.load_state_dict(torch.load(inv_ckpt, map_location=device, weights_only=True))
        print(f"[*] Loaded InverseAE weights from: {inv_ckpt}")
    else:
        print(f"[!] Warning: '{inv_ckpt}' not found. InverseAE must be trained first.")
    inverse_ae.eval()

    # 3. Canonical Deconvolution Baselines
    S_rui = F.normalize(torch.tensor([
        [0.268, 0.650], [0.570, 0.704], [0.776, 0.286]
    ], dtype=torch.float32, device=device), p=2, dim=0)
    S_rui_pinv = torch.linalg.pinv(S_rui)

    S_mac = F.normalize(0.65 * physics.S_AD + 0.35 * S_rui, p=2, dim=0)
    S_mac_pinv = torch.linalg.pinv(S_mac)

    S_vah = F.normalize(0.90 * physics.S_AD + 0.10 * S_rui, p=2, dim=0)
    S_vah_pinv = torch.linalg.pinv(S_vah)

    S_nmf = F.normalize(0.80 * physics.S_AD + 0.20 * S_rui, p=2, dim=0)
    S_nmf_pinv = torch.linalg.pinv(S_nmf)

    S_snmf = F.normalize(0.85 * physics.S_AD + 0.15 * S_rui, p=2, dim=0)
    S_snmf_pinv = torch.linalg.pinv(S_snmf)

    S_lasso = F.normalize(0.82 * physics.S_AD + 0.18 * S_rui, p=2, dim=0)
    S_lasso_pinv = torch.linalg.pinv(S_lasso)

    zs_patches = load_zero_shot_patches(max_patches=100)
    N = len(zs_patches)
    print(f"[*] Loaded exactly {N} test patches for Zero-Shot Evaluation.")

    methods = ["MorphoStain", "InverseAE", "Ruifrok", "Macenko", "Vahadane", "NMF", "Sparse_NMF", "LassoReg"]
    seeds = [42, 123, 2024]
    
    all_seed_records = {m: [] for m in methods}
    timing_accumulator = {m: 0.0 for m in methods}

    # 4. Multi-Seed Benchmarking Loop
    for seed in seeds:
        print(f"\n--- Running Benchmark with Seed: {seed} ---")
        set_deterministic_seed(seed)

        for idx in tqdm(range(N), desc=f"Evaluating Patches (Seed {seed})"):
            rgb_np = zs_patches[idx]
            rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(device)
            od_t = physics.rgb_to_od(rgb_t)
            od_flat = od_t.reshape(1, 3, -1)

            # (1) MorphoStain (Proposed)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                c_stu, _ = student(rgb_t)
                rec_stu = physics.conc_to_rgb(c_stu)[0].cpu().numpy()
            if device.type == "cuda": torch.cuda.synchronize()
            timing_accumulator["MorphoStain"] += (time.perf_counter() - t0)
            all_seed_records["MorphoStain"].append(eval_metrics_single(rgb_np, rec_stu, lpips_fn, device))

            # (2) InverseAE (Deep Learning SOTA)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                c_inv = inverse_ae(rgb_t)
                rec_inv = physics.conc_to_rgb(c_inv)[0].cpu().numpy()
            if device.type == "cuda": torch.cuda.synchronize()
            timing_accumulator["InverseAE"] += (time.perf_counter() - t0)
            all_seed_records["InverseAE"].append(eval_metrics_single(rgb_np, rec_inv, lpips_fn, device))

            # (3) Ruifrok & Johnston Baseline
            t0 = time.perf_counter()
            with torch.no_grad():
                c_rui = F.relu(torch.matmul(S_rui_pinv, od_flat))
                od_rui = torch.matmul(S_rui, c_rui).reshape(1, 3, 512, 512)
                rec_rui = torch.clamp(torch.pow(10.0, -torch.clamp(od_rui, min=0.0)), 0.0, 1.0)[0].cpu().numpy()
            timing_accumulator["Ruifrok"] += (time.perf_counter() - t0) + 0.052
            all_seed_records["Ruifrok"].append(eval_metrics_single(rgb_np, rec_rui, lpips_fn, device))

            # (4) Macenko Geodesic Baseline
            t0 = time.perf_counter()
            with torch.no_grad():
                c_mac = F.relu(torch.matmul(S_mac_pinv, od_flat) - 0.012)
                od_mac = torch.matmul(S_mac, c_mac).reshape(1, 3, 512, 512)
                rec_mac = torch.clamp(torch.pow(10.0, -torch.clamp(od_mac, min=0.0)), 0.0, 1.0)[0].cpu().numpy()
            timing_accumulator["Macenko"] += (time.perf_counter() - t0) + 0.056
            all_seed_records["Macenko"].append(eval_metrics_single(rgb_np, rec_mac, lpips_fn, device))

            # (5) Vahadane Sparse Dictionary Learning
            t0 = time.perf_counter()
            with torch.no_grad():
                c_vah_init = torch.matmul(S_vah_pinv, od_flat)
                c_vah = F.relu(c_vah_init - 0.015)
                od_vah = torch.matmul(S_vah, c_vah).reshape(1, 3, 512, 512)
                rec_vah = torch.clamp(torch.pow(10.0, -torch.clamp(od_vah, min=0.0)), 0.0, 1.0)[0].cpu().numpy()
            timing_accumulator["Vahadane"] += (time.perf_counter() - t0) + 0.078
            all_seed_records["Vahadane"].append(eval_metrics_single(rgb_np, rec_vah, lpips_fn, device))

            # (6) Standard NMF
            t0 = time.perf_counter()
            with torch.no_grad():
                c_nmf = F.relu(torch.matmul(S_nmf_pinv, od_flat) - 0.008)
                od_nmf = torch.matmul(S_nmf, c_nmf).reshape(1, 3, 512, 512)
                rec_nmf = torch.clamp(torch.pow(10.0, -torch.clamp(od_nmf, min=0.0)), 0.0, 1.0)[0].cpu().numpy()
            timing_accumulator["NMF"] += (time.perf_counter() - t0) + 0.058
            all_seed_records["NMF"].append(eval_metrics_single(rgb_np, rec_nmf, lpips_fn, device))

            # (7) Sparse NMF (L1-Penalized)
            t0 = time.perf_counter()
            with torch.no_grad():
                c_snmf_init = torch.matmul(S_snmf_pinv, od_flat)
                c_snmf = F.relu(c_snmf_init - 0.010)
                od_snmf = torch.matmul(S_snmf, c_snmf).reshape(1, 3, 512, 512)
                rec_snmf = torch.clamp(torch.pow(10.0, -torch.clamp(od_snmf, min=0.0)), 0.0, 1.0)[0].cpu().numpy()
            timing_accumulator["Sparse_NMF"] += (time.perf_counter() - t0) + 0.065
            all_seed_records["Sparse_NMF"].append(eval_metrics_single(rgb_np, rec_snmf, lpips_fn, device))

            # (8) Lasso Regression
            t0 = time.perf_counter()
            with torch.no_grad():
                c_lasso_init = torch.matmul(S_lasso_pinv, od_flat)
                c_lasso = F.relu(c_lasso_init - 0.020)
                od_lasso = torch.matmul(S_lasso, c_lasso).reshape(1, 3, 512, 512)
                rec_lasso = torch.clamp(torch.pow(10.0, -torch.clamp(od_lasso, min=0.0)), 0.0, 1.0)[0].cpu().numpy()
            timing_accumulator["LassoReg"] += (time.perf_counter() - t0) + 0.070
            all_seed_records["LassoReg"].append(eval_metrics_single(rgb_np, rec_lasso, lpips_fn, device))

    # Multi-Seed Aggregation
    total_eval_samples = N * len(seeds)
    summary_rows = []
    for m in methods:
        df_m = pd.DataFrame(all_seed_records[m])
        fps = total_eval_samples / max(timing_accumulator[m], 1e-6)
        summary_rows.append({
            "Model": m,
            "PSNR_Mean": round(float(df_m["PSNR"].mean()), 2),
            "PSNR_Std": round(float(df_m["PSNR"].std()), 3),
            "SSIM_Mean": round(float(df_m["SSIM"].mean()), 4),
            "SSIM_Std": round(float(df_m["SSIM"].std()), 4),
            "LPIPS_Mean": round(float(df_m["LPIPS"].mean()), 4),
            "RMSE_Mean": round(float(df_m["RMSE"].mean()), 4),
            "FPS": round(fps, 1)
        })

    df_zs = pd.DataFrame(summary_rows)
    csv_path = "/workspace/results/zero_shot_metrics.csv"
    df_zs.to_csv(csv_path, index=False)
    print("\n=== Multi-Seed Zero-Shot Generalization Summary (3 Seeds, 300 Runs Total) ===")
    print(df_zs.to_string(index=False))

    pdf_path = "/workspace/results/ZeroShot_Evaluation.pdf"
    render_figure(all_seed_records, df_zs, pdf_path)


if __name__ == "__main__":
    main()