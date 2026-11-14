"""
Balanced Cohort Evaluation Module (Amyloid-Beta vs. Tau AT8 Subgroups).

Evaluates MorphoStain student reconstruction fidelity (PSNR, SSIM) on a strictly
balanced held-out sub-cohort, generating summary statistics for publication tables.
Cross-platform compatible across macOS local runs, Docker containers, and RunPod instances.
"""

import os
from pathlib import Path
from importlib import import_module
import h5py
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# =========================================================================
# PATH CONFIGURATION 
# =========================================================================
CURRENT_FILE = Path(__file__).resolve()
# Resolves root directory whether script is executed from root or inside code/
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "code" else CURRENT_FILE.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset path resolution
H5_PATH = PROJECT_ROOT / "latent_test.h5"
if not H5_PATH.exists():
    H5_PATH = Path("/workspace/latent_test.h5")

# Model weights path resolution
WEIGHTS_PATH = RESULTS_DIR / "best_student.pth"
if not WEIGHTS_PATH.exists():
    WEIGHTS_PATH = Path("/workspace/results/best_student.pth")


def evaluate_balanced_cohort(n_samples_per_class: int = 100):
    """
    Evaluates MorphoStain on an equal number of Amyloid-Beta and Tau patches.
    
    Args:
        n_samples_per_class: Number of representative patches to evaluate per biomarker class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing evaluation pipeline on device: {device}")

    # Dynamic imports of project modules
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

    if not WEIGHTS_PATH.exists():
        print(f"[!] Student model weights not found at: {WEIGHTS_PATH}. Aborting.")
        return

    student = FastStudentUNet().to(device)
    student.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    )
    student.eval()
    physics = BeerLambertPhysics().to(device)

    if not H5_PATH.exists():
        print(f"[!] H5 dataset not found at: {H5_PATH}. Aborting.")
        return

    # Memory-efficient indexing without loading entire RGB array into RAM
    with h5py.File(H5_PATH, "r") as f:
        total_patches = f["rgb"].shape[0]
        if "slide_label" in f:
            labels = f["slide_label"][:]
        else:
            print("[!] Warning: 'slide_label' key not found in HDF5. Defaulting to index split.")
            labels = np.zeros(total_patches)

        amyloid_indices = np.where(labels < 0.5)[0]
        tau_indices = np.where(labels >= 0.5)[0]

        print(f"Total cohort availability - Amyloid (6E10): {len(amyloid_indices)} | Tau (AT8): {len(tau_indices)}")

        n_per_class = min(n_samples_per_class, len(amyloid_indices), len(tau_indices))

        np.random.seed(42)
        selected_amyloid = np.random.choice(amyloid_indices, n_per_class, replace=False)
        selected_tau = np.random.choice(tau_indices, n_per_class, replace=False)

        balanced_indices = np.concatenate([selected_amyloid, selected_tau])
        np.random.shuffle(balanced_indices)

        results = []
        print(f"[*] Evaluating balanced cohort: {len(balanced_indices)} patches ({n_per_class} Amyloid, {n_per_class} Tau)...")

        with torch.no_grad():
            for i, idx in enumerate(balanced_indices):
                # Lazy slice retrieval from disk prevents RAM exhaustion
                rgb_np = f["rgb"][idx]
                label = labels[idx]
                cohort_name = "Tau (AT8)" if label >= 0.5 else "Amyloid-Beta (6E10)"

                rgb_t = torch.from_numpy(rgb_np).unsqueeze(0).to(device)
                
                # Robust unpack supporting both forward() and forward_train() signatures
                model_out = student(rgb_t)
                c_mean = model_out[0] if isinstance(model_out, (tuple, list)) else model_out
                
                recon_t = physics.conc_to_rgb(c_mean)

                true_c = np.clip(rgb_np, 0.0, 1.0)
                pred_c = np.clip(recon_t.squeeze(0).cpu().numpy(), 0.0, 1.0)

                # Compute quantitative metrics
                psnr_val = float(psnr_fn(true_c, pred_c, data_range=1.0))
                ssim_val = float(ssim_fn(
                    np.transpose(true_c, (1, 2, 0)),
                    np.transpose(pred_c, (1, 2, 0)),
                    data_range=1.0,
                    channel_axis=2
                ))

                results.append({
                    "Patch_ID": int(idx),
                    "Subgroup": cohort_name,
                    "PSNR": psnr_val,
                    "SSIM": ssim_val
                })

                if (i + 1) % 25 == 0 or (i + 1) == len(balanced_indices):
                    print(f"  -> Processed [{i + 1}/{len(balanced_indices)}] patches...")

    df_res = pd.DataFrame(results)

    # Compute descriptive summary metrics
    summary = df_res.groupby("Subgroup")[["PSNR", "SSIM"]].agg(["mean", "std", "count"])
    print("\n=== Balanced Evaluation Summary ===")
    print(summary)

    # Export statistical summary report
    report_path = RESULTS_DIR / "balanced_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as out:
        out.write("Balanced Cohort Evaluation Report (Equal Amyloid vs. Tau)\n")
        out.write("=" * 60 + "\n\n")
        out.write(summary.to_string())
        out.write("\n")

    # Export raw per-patch metric logs
    csv_path = RESULTS_DIR / "balanced_subgroup_metrics.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n[✓] Results successfully saved to:")
    print(f"    - Report: {report_path}")
    print(f"    - Metrics CSV: {csv_path}")


if __name__ == "__main__":
    evaluate_balanced_cohort()