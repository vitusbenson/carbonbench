"""
Plot SDE posterior sampling ablation results (Phase 7).

Reads ablation_summary.json and produces:
  1. RMSE + spread vs sigma_max (dual y-axis)
  2. Spread-skill ratio vs sigma_max (target line at 1.0)
  3. Noise schedule comparison (bar chart)
  4. Corrector steps sweep (RMSE + spread vs n_corrector)
  5. ODE vs SDE comparison (FlowDPS vs best SDE, bar chart)
  6. Summary bar chart of all configs

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


def _get_metric(metrics, key="co2massmix_delta_rmse_xco2"):
    if isinstance(metrics, dict) and "error" not in metrics:
        return metrics.get(key, float("nan"))
    return float("nan")


def plot_rmse_spread_vs_sigma_max(results, out_dir):
    """RMSE + spread vs sigma_max (dual y-axis)."""
    sigmas = []
    rmse_3d = []
    spread_3d = []

    for name, metrics in sorted(results.items()):
        if not name.startswith("sde_smax"):
            continue
        s = float(name.split("smax")[1])
        sigmas.append(s)
        rmse_3d.append(_get_metric(metrics, "co2massmix_delta_rmse"))
        spread_3d.append(_get_metric(metrics, "co2massmix_delta_spread_3d"))

    if not sigmas:
        print("No sigma_max sweep configs found, skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(sigmas, rmse_3d, "o-", label="RMSE 3D", color="tab:blue")
    ax2.plot(sigmas, spread_3d, "s--", label="Spread 3D", color="tab:orange")

    ax1.set_xlabel("sigma_max")
    ax1.set_ylabel("RMSE (ppm)", color="tab:blue")
    ax2.set_ylabel("Spread (ppm)", color="tab:orange")
    ax1.set_xscale("log")
    ax1.set_title("SDE: RMSE and Spread vs sigma_max")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)

    fig.savefig(out_dir / "rmse_spread_vs_sigma_max.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'rmse_spread_vs_sigma_max.png'}")


def plot_ssr_vs_sigma_max(results, out_dir):
    """Spread-skill ratio vs sigma_max with target line at 1.0."""
    sigmas = []
    ssr = []

    for name, metrics in sorted(results.items()):
        if not name.startswith("sde_smax"):
            continue
        s = float(name.split("smax")[1])
        sigmas.append(s)
        ssr.append(_get_metric(metrics, "co2massmix_delta_spread_skill_ratio"))

    if not sigmas:
        print("No sigma_max sweep configs found, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, ssr, "o-", label="SSR", color="tab:green", markersize=8)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Target (SSR=1.0)")
    ax.set_xlabel("sigma_max")
    ax.set_ylabel("Spread-Skill Ratio")
    ax.set_xscale("log")
    ax.set_title("SDE: Spread-Skill Ratio vs sigma_max")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / "ssr_vs_sigma_max.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'ssr_vs_sigma_max.png'}")


def plot_noise_schedule_comparison(results, out_dir):
    """Bar chart: noise schedule comparison."""
    schedules = ["annealed", "constant", "cosine"]
    rmse_3d = []
    spread_3d = []
    valid_schedules = []

    for sched in schedules:
        key = f"sde_sched_{sched}"
        if key in results:
            valid_schedules.append(sched)
            rmse_3d.append(_get_metric(results[key], "co2massmix_delta_rmse"))
            spread_3d.append(_get_metric(results[key], "co2massmix_delta_spread_3d"))

    if not valid_schedules:
        print("No noise schedule configs found, skipping plot.")
        return

    x = np.arange(len(valid_schedules))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, rmse_3d, width, label="RMSE 3D", color="tab:blue")
    ax.bar(x + width / 2, spread_3d, width, label="Spread 3D", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_schedules)
    ax.set_ylabel("ppm")
    ax.set_title("SDE: Noise Schedule Comparison (sigma_max=0.3)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(out_dir / "noise_schedule_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'noise_schedule_comparison.png'}")


def plot_corrector_sweep(results, out_dir):
    """RMSE + spread vs n_corrector_steps."""
    n_corr = []
    rmse_3d = []
    spread_3d = []

    # Include 0 corrector = sde_smax0.3
    if "sde_smax0.3" in results:
        n_corr.append(0)
        rmse_3d.append(_get_metric(results["sde_smax0.3"], "co2massmix_delta_rmse"))
        spread_3d.append(_get_metric(results["sde_smax0.3"], "co2massmix_delta_spread_3d"))

    for name, metrics in sorted(results.items()):
        if not name.startswith("pc_c") or "eps" in name:
            continue
        k = int(name.split("_c")[1].split("_")[0])
        n_corr.append(k)
        rmse_3d.append(_get_metric(metrics, "co2massmix_delta_rmse"))
        spread_3d.append(_get_metric(metrics, "co2massmix_delta_spread_3d"))

    if len(n_corr) < 2:
        print("Not enough corrector sweep configs found, skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(n_corr, rmse_3d, "o-", label="RMSE 3D", color="tab:blue")
    ax2.plot(n_corr, spread_3d, "s--", label="Spread 3D", color="tab:orange")

    ax1.set_xlabel("Number of Corrector Steps")
    ax1.set_ylabel("RMSE (ppm)", color="tab:blue")
    ax2.set_ylabel("Spread (ppm)", color="tab:orange")
    ax1.set_title("SDE: Corrector Steps Sweep (sigma_max=0.3)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)

    fig.savefig(out_dir / "corrector_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'corrector_sweep.png'}")


def plot_ode_vs_sde(results, out_dir):
    """ODE vs SDE comparison (FlowDPS vs best SDE, bar chart)."""
    compare_keys = ["unconditional", "best_flowdps", "sde_smax0.3"]
    names = []
    rmse_xco2 = []
    spread_3d = []

    for key in compare_keys:
        if key in results:
            names.append(key)
            rmse_xco2.append(_get_metric(results[key], "co2massmix_delta_rmse_xco2"))
            spread_3d.append(_get_metric(results[key], "co2massmix_delta_spread_3d"))

    if not names:
        print("No comparison configs found, skipping plot.")
        return

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, rmse_xco2, width, label="RMSE XCO2", color="tab:blue")
    ax.bar(x + width / 2, spread_3d, width, label="Spread 3D", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("ppm")
    ax.set_title("ODE (FlowDPS) vs SDE Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(out_dir / "ode_vs_sde.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'ode_vs_sde.png'}")


def plot_summary_table(results, out_dir):
    """Summary bar chart of all configurations."""
    valid = {k: v for k, v in results.items() if isinstance(v, dict) and "error" not in v}
    if not valid:
        print("No valid results to plot.")
        return

    names = list(valid.keys())
    rmse_xco2 = [_get_metric(v, "co2massmix_delta_rmse_xco2") for v in valid.values()]
    spread_3d = [_get_metric(v, "co2massmix_delta_spread_3d") for v in valid.values()]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # RMSE bar chart
    colors = []
    for n in names:
        if n == "unconditional":
            colors.append("tab:gray")
        elif n == "best_flowdps":
            colors.append("tab:red")
        elif "pc_" in n:
            colors.append("tab:green")
        else:
            colors.append("tab:blue")

    axes[0].barh(range(len(names)), rmse_xco2, color=colors)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel("RMSE XCO2 (ppm)")
    axes[0].set_title("RMSE XCO2")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis="x")

    # Spread bar chart
    axes[1].barh(range(len(names)), spread_3d, color=colors)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel("Spread 3D (ppm)")
    axes[1].set_title("Ensemble Spread 3D")
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.suptitle("SDE Ablation: All Configurations", fontsize=14)
    plt.tight_layout()

    fig.savefig(out_dir / "ablation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'ablation_summary.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot SDE ablation results")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    if args.results_dir is None:
        results_dir = Path(__file__).resolve().parent / "results"
    else:
        results_dir = Path(args.results_dir)

    results = load_results(results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_rmse_spread_vs_sigma_max(results, out_dir)
    plot_ssr_vs_sigma_max(results, out_dir)
    plot_noise_schedule_comparison(results, out_dir)
    plot_corrector_sweep(results, out_dir)
    plot_ode_vs_sde(results, out_dir)
    plot_summary_table(results, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
