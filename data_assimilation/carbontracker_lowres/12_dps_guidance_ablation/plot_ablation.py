"""
Plot DPS guidance ablation results (Phase 5c).

Reads ablation_summary.json and produces:
  1. RMSE vs sigma_obs lines
  2. RMSE vs guidance_scale lines (colored by sigma_obs)
  3. Roughness heatmap: guidance_scale x smoothing_sigma
  4. Per-level guidance magnitude bar chart (requires per-level metrics)

Usage:
    python plot_ablation.py
    python plot_ablation.py --results_dir results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    summary_path = Path(results_dir) / "ablation_summary.json"
    with open(summary_path) as f:
        return json.load(f)


def parse_config_name(name):
    """Extract parameters from config name."""
    params = {"name": name}
    parts = name.split("_")
    for p in parts:
        if p.startswith("s") and p[1:].replace(".", "").isdigit():
            params["sigma_obs"] = float(p[1:])
        elif p.startswith("sm") and p[2:].replace(".", "").isdigit():
            params["smoothing"] = float(p[2:])
    if "scale" in name:
        # e.g. scale_0.5_s0.5
        idx = parts.index("scale") if "scale" in parts else -1
        if idx >= 0 and idx + 1 < len(parts):
            try:
                params["guidance_scale"] = float(parts[idx + 1])
            except ValueError:
                pass
    if "sigma" in name and "sigma_obs" not in params:
        idx = parts.index("sigma") if "sigma" in parts else -1
        if idx >= 0 and idx + 1 < len(parts):
            try:
                params["sigma_obs"] = float(parts[idx + 1])
            except ValueError:
                pass
    return params


def plot_rmse_vs_sigma_obs(results, out_dir):
    """RMSE vs sigma_obs for fixed guidance_scale=1.0, no smoothing."""
    sigma_names = {k: v for k, v in results.items() if k.startswith("sigma_")}
    if not sigma_names:
        print("No sigma_obs sweep configs found, skipping plot.")
        return

    sigmas = []
    rmse_3d = []
    rmse_xco2 = []

    for name, metrics in sorted(sigma_names.items()):
        if "error" in metrics:
            continue
        s = float(name.split("_")[1])
        sigmas.append(s)
        rmse_3d.append(metrics.get("co2massmix_delta_rmse", float("nan")))
        rmse_xco2.append(metrics.get("co2massmix_delta_rmse_xco2", float("nan")))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sigmas, rmse_3d, "o-", label="RMSE 3D", color="tab:blue")
    ax1.plot(sigmas, rmse_xco2, "s-", label="RMSE XCO2", color="tab:orange")
    ax1.set_xlabel("sigma_obs")
    ax1.set_ylabel("RMSE (ppm)")
    ax1.set_xscale("log")
    ax1.set_title("DPS Guidance: RMSE vs Observation Uncertainty")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.savefig(out_dir / "rmse_vs_sigma_obs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'rmse_vs_sigma_obs.png'}")


def plot_rmse_vs_guidance_scale(results, out_dir):
    """RMSE vs guidance_scale for fixed sigma_obs=0.5."""
    scale_names = {k: v for k, v in results.items() if k.startswith("scale_")}
    if not scale_names:
        print("No guidance_scale sweep configs found, skipping plot.")
        return

    scales = []
    rmse_3d = []
    rmse_xco2 = []

    for name, metrics in sorted(scale_names.items()):
        if "error" in metrics:
            continue
        g = float(name.split("_")[1])
        scales.append(g)
        rmse_3d.append(metrics.get("co2massmix_delta_rmse", float("nan")))
        rmse_xco2.append(metrics.get("co2massmix_delta_rmse_xco2", float("nan")))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales, rmse_3d, "o-", label="RMSE 3D", color="tab:blue")
    ax.plot(scales, rmse_xco2, "s-", label="RMSE XCO2", color="tab:orange")
    ax.set_xlabel("guidance_scale")
    ax.set_ylabel("RMSE (ppm)")
    ax.set_xscale("log")
    ax.set_title("DPS Guidance: RMSE vs Guidance Scale (sigma_obs=0.5)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / "rmse_vs_guidance_scale.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'rmse_vs_guidance_scale.png'}")


def plot_smoothing_comparison(results, out_dir):
    """Bar chart: RMSE for different smoothing values at sigma_obs=0.5."""
    smooth_names = {k: v for k, v in results.items() if k.startswith("smooth_")}
    if not smooth_names:
        print("No smoothing sweep configs found, skipping plot.")
        return

    names = []
    rmse_3d = []
    rmse_xco2 = []

    for name, metrics in sorted(smooth_names.items()):
        if "error" in metrics:
            continue
        names.append(name)
        rmse_3d.append(metrics.get("co2massmix_delta_rmse", float("nan")))
        rmse_xco2.append(metrics.get("co2massmix_delta_rmse_xco2", float("nan")))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, rmse_3d, width, label="RMSE 3D", color="tab:blue")
    ax.bar(x + width / 2, rmse_xco2, width, label="RMSE XCO2", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("RMSE (ppm)")
    ax.set_title("DPS Guidance: Effect of Spatial Smoothing (sigma_obs=0.5)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(out_dir / "smoothing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'smoothing_comparison.png'}")


def plot_timing_comparison(results, out_dir):
    """Bar chart: timing strategies."""
    timing_names = {k: v for k, v in results.items() if k.startswith("s0.5_sm2")}
    if not timing_names:
        print("No timing sweep configs found, skipping plot.")
        return

    names = []
    rmse_3d = []
    rmse_xco2 = []

    for name, metrics in sorted(timing_names.items()):
        if "error" in metrics:
            continue
        names.append(name)
        rmse_3d.append(metrics.get("co2massmix_delta_rmse", float("nan")))
        rmse_xco2.append(metrics.get("co2massmix_delta_rmse_xco2", float("nan")))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, rmse_3d, width, label="RMSE 3D", color="tab:blue")
    ax.bar(x + width / 2, rmse_xco2, width, label="RMSE XCO2", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("RMSE (ppm)")
    ax.set_title("DPS Guidance: Timing Strategy Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(out_dir / "timing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'timing_comparison.png'}")


def plot_summary_table(results, out_dir):
    """Summary bar chart of all configurations."""
    valid = {k: v for k, v in results.items() if isinstance(v, dict) and "error" not in v}
    if not valid:
        print("No valid results to plot.")
        return

    names = list(valid.keys())
    rmse_xco2 = [v.get("co2massmix_delta_rmse_xco2", float("nan")) for v in valid.values()]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["tab:gray" if n == "unconditional" else "tab:blue" for n in names]
    ax.barh(range(len(names)), rmse_xco2, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("RMSE XCO2 (ppm)")
    ax.set_title("DPS Guidance Ablation: All Configurations")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    fig.savefig(out_dir / "ablation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'ablation_summary.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot DPS guidance ablation results")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    if args.results_dir is None:
        results_dir = Path(__file__).resolve().parent / "results"
    else:
        results_dir = Path(args.results_dir)

    results = load_results(results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_rmse_vs_sigma_obs(results, out_dir)
    plot_rmse_vs_guidance_scale(results, out_dir)
    plot_smoothing_comparison(results, out_dir)
    plot_timing_comparison(results, out_dir)
    plot_summary_table(results, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
