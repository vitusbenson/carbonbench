"""Plot DPS guidance ablation results (Phase 5c).

Custom plots: sigma_obs sweep, guidance_scale sweep, smoothing, timing.
Standard plots via AblationRunner.plot_results().
"""

import argparse
from pathlib import Path

from neural_transport.plots.metrics_plots import load_ablation_results, plot_ablation_sweep, plot_summary_bars


def plot_rmse_vs_sigma_obs(results, out_dir):
    """RMSE vs sigma_obs for fixed guidance_scale=1.0."""
    sigma_configs = {k: v for k, v in results.items() if k.startswith("sigma_")}
    if sigma_configs:
        plot_ablation_sweep(
            sigma_configs, "RMSE_3D_co2molemix",
            lambda name: float(name.split("_")[1]),
            out_dir, xlabel="sigma_obs", log_x=True,
            title="DPS: RMSE vs Observation Uncertainty",
            second_metric_key="RMSE_lat_co2molemix", second_ylabel="RMSE lat",
        )


def plot_rmse_vs_guidance_scale(results, out_dir):
    """RMSE vs guidance_scale for sigma_obs=0.5."""
    scale_configs = {k: v for k, v in results.items() if k.startswith("scale_")}
    if scale_configs:
        plot_ablation_sweep(
            scale_configs, "RMSE_3D_co2molemix",
            lambda name: float(name.split("_")[1]),
            out_dir, xlabel="guidance_scale", log_x=True,
            title="DPS: RMSE vs Guidance Scale (sigma_obs=0.5)",
        )


def plot_smoothing_comparison(results, out_dir):
    """Bar chart for smoothing sweep."""
    smooth_configs = {k: v for k, v in results.items() if k.startswith("smooth_")}
    if smooth_configs:
        plot_summary_bars(smooth_configs, "RMSE_3D_co2molemix", out_dir,
                          title="DPS: Effect of Spatial Smoothing")


def plot_timing_comparison(results, out_dir):
    """Bar chart for timing strategies."""
    timing_configs = {k: v for k, v in results.items() if k.startswith("s0.5_sm2")}
    if timing_configs:
        plot_summary_bars(timing_configs, "RMSE_3D_co2molemix", out_dir,
                          title="DPS: Timing Strategy Comparison")


CUSTOM_PLOTS = [plot_rmse_vs_sigma_obs, plot_rmse_vs_guidance_scale,
                plot_smoothing_comparison, plot_timing_comparison]


def main():
    parser = argparse.ArgumentParser(description="Plot DPS guidance ablation results")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parent / "results"
    results = load_ablation_results(results_dir)

    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for plot_fn in CUSTOM_PLOTS:
        plot_fn(results, out_dir)
    plot_summary_bars(results, "RMSE_lat_co2molemix", out_dir, title="DPS Ablation: All Configs")
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
