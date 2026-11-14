"""
Phase 8H: Comprehensive Statistical Significance & Hypothesis Testing Suite.
Dynamically computes parametric and non-parametric statistical tests across the test set
for all baseline models present in psnr_distributions.csv against MorphoStain (Student_PSNR).

Saves:
- /workspace/results/benchmark_statistical_tests.csv
- /workspace/results/statistical_summary_table.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats


def compute_cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Computes paired Cohen's d effect size for matched patch evaluations."""
    diff = x - y
    std_diff = float(np.std(diff, ddof=1))
    if std_diff < 1e-12:
        return 0.0
    return float(np.mean(diff) / std_diff)


def format_p_value(p: float) -> str:
    """Formats p-values cleanly for publication tables."""
    if p < 1e-16:
        return "< 1e-16 (***)"
    if p < 1e-4:
        return f"{p:.2e} (***)"
    if p < 1e-2:
        return f"{p:.4f} (**)"
    if p < 0.05:
        return f"{p:.4f} (*)"
    return f"{p:.4f} (ns)"


def main():
    print("=" * 85)
    print(" PHASE 8H: DYNAMIC STATISTICAL SIGNIFICANCE & EFFECT SIZE ANALYSIS")
    print("=" * 85)

    os.makedirs("/workspace/results", exist_ok=True)
    dist_csv = "/workspace/results/psnr_distributions.csv"
    patch_csv = "/workspace/results/per_patch_test_metrics.csv"

    if not os.path.exists(dist_csv):
        print(f"[!] Warning: '{dist_csv}' not found. Run evaluateBaselines.py first.")
        return

    df_dist = pd.read_csv(dist_csv)
    
    # Ensure the primary target model (MorphoStain) exists
    if "Student_PSNR" not in df_dist.columns:
        print("[!] Error: 'Student_PSNR' column not found in distribution CSV.")
        return

    student_psnr = df_dist["Student_PSNR"].to_numpy(dtype=np.float64)
    n_samples = len(student_psnr)

    # Dynamically find all other models ending in _PSNR
    baseline_cols = [col for col in df_dist.columns if col.endswith("_PSNR") and col != "Student_PSNR"]
    
    if not baseline_cols:
        print("[!] Warning: No baseline PSNR columns found to compare against.")
        return
        
    print(f"[*] Found {len(baseline_cols)} baselines to compare: {', '.join([c.replace('_PSNR', '') for c in baseline_cols])}")

    stat_rows = []
    for col_name in baseline_cols:
        model_name = col_name.replace("_PSNR", "")
        comp_label = f"MorphoStain vs {model_name}"
        
        base_psnr = df_dist[col_name].to_numpy(dtype=np.float64)
        diff = student_psnr - base_psnr

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        se_diff = std_diff / np.sqrt(n_samples)
        ci95_low = mean_diff - 1.96 * se_diff
        ci95_high = mean_diff + 1.96 * se_diff

        # Paired t-test
        t_stat, p_ttest = stats.ttest_rel(student_psnr, base_psnr)

        # Wilcoxon signed-rank test (handle zero differences safely)
        if np.allclose(diff, 0.0):
            w_stat, p_wilcox = 0.0, 1.0
        else:
            w_stat, p_wilcox = stats.wilcoxon(student_psnr, base_psnr, zero_method="zsplit")

        # Mann-Whitney U test
        u_stat, p_mwu = stats.mannwhitneyu(student_psnr, base_psnr, alternative="two-sided")

        cohens_d = compute_cohens_d_paired(student_psnr, base_psnr)

        stat_rows.append({
            "Comparison": comp_label,
            "N_Patches": n_samples,
            "PRISM_Mean_dB": round(float(np.mean(student_psnr)), 2),
            "Baseline_Mean_dB": round(float(np.mean(base_psnr)), 2),
            "Mean_Gain_dB": round(mean_diff, 2),
            "95%_CI_dB": f"[{ci95_low:+.2f}, {ci95_high:+.2f}]",
            "t_Statistic": round(float(t_stat), 2),
            "Paired_t_p_value": format_p_value(float(p_ttest)),
            "Wilcoxon_p_value": format_p_value(float(p_wilcox)),
            "MannWhitney_p_value": format_p_value(float(p_mwu)),
            "Cohens_d": round(cohens_d, 2)
        })

    df_tests = pd.DataFrame(stat_rows)
    tests_out = "/workspace/results/benchmark_statistical_tests.csv"
    df_tests.to_csv(tests_out, index=False)

    print("\n--- Paired Hypothesis Tests (Test Split, N = {}) ---".format(n_samples))
    print(df_tests.to_string(index=False))

    # Detailed Distributional Summary of MorphoStain Test Metrics
    if os.path.exists(patch_csv):
        df_patch = pd.read_csv(patch_csv)
        metric_cols = ["PSNR", "SSIM", "MSE", "RMSE", "Cosine_Sim", "LPIPS"]
        summary_rows = []

        for m in metric_cols:
            if m not in df_patch.columns:
                continue
            vals = df_patch[m].dropna().to_numpy(dtype=np.float64)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            se_v = std_v / np.sqrt(len(vals))
            med_v = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25.0, 75.0])
            iqr_v = float(q75 - q25)

            summary_rows.append({
                "Metric": m,
                "Mean": round(mean_v, 5),
                "Std": round(std_v, 5),
                "95%_CI_Lower": round(mean_v - 1.96 * se_v, 5),
                "95%_CI_Upper": round(mean_v + 1.96 * se_v, 5),
                "Median": round(med_v, 5),
                "IQR": round(iqr_v, 5),
                "Min": round(float(np.min(vals)), 5),
                "Max": round(float(np.max(vals)), 5)
            })

        df_summary = pd.DataFrame(summary_rows)
        summary_out = "/workspace/results/statistical_summary_table.csv"
        df_summary.to_csv(summary_out, index=False)
        print("\n--- MorphoStain Per-Patch Distributional Statistics ---")
        print(df_summary.to_string(index=False))

    print("\n" + "=" * 85)
    print(f"[✓] Saved dynamic statistical significance reports to: {tests_out}")
    print("=" * 85)


if __name__ == "__main__":
    main()