"""Plot FlowDPS ablation results (Phase 6).

Custom plots: sigma_obs sweep, step convergence, smoothing, FlowDPS vs DPS.
"""

import argparse
from pathlib import Path

from neural_transport.plots.metrics_plots import load_ablation_results, plot_ablation_sweep, plot_summary_bars


def plot_rmse_vs_sigma_obs(results, out_dir):
    sigma_configs = {k: v for k, v in results.items()
                     if k.startswith("flowdps_s") and "smooth" not in k
                     and "steps" not in k and "fixed" not in k}
    if sigma_configs:
        plot_ablation_sweep(
            sigma_configs, "RMSE_3D_co2molemix",
            lambda name: float(name.split("_s")[1]),
            out_dir, xlabel="sigma_obs", log_x=True,
            title="FlowDPS: RMSE vs Observation Uncertainty",
            second_metric_key="RMSE_lat_co2molemix", second_ylabel="RMSE lat",
        )


def plot_rmse_vs_steps(results, out_dir):
    step_configs = {k: v for k, v in results.items() if "steps" in k}
    if step_configs:
        plot_ablation_sweep(
            step_configs, "RMSE_3D_co2molemix",
            lambda name: int(name.split("steps")[1]),
            out_dir, xlabel="Number of Steps",
            title="FlowDPS: Step Count Convergence",
        )


def plot_flowdps_vs_dps(results, out_dir):
    compare = {k: v for k, v in results.items() if k in ("unconditional", "best_dps", "flowdps_s0.1")}
    if compare:
        plot_summary_bars(compare, "RMSE_lat_co2molemix", out_dir,
                          title="FlowDPS vs DPS Guidance Comparison")


CUSTOM_PLOTS = [plot_rmse_vs_sigma_obs, plot_rmse_vs_steps, plot_flowdps_vs_dps]


def main():
    parser = argparse.ArgumentParser(description="Plot FlowDPS ablation results")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parent / "results"
    results = load_ablation_results(results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for plot_fn in CUSTOM_PLOTS:
        plot_fn(results, out_dir)
    plot_summary_bars(results, "RMSE_lat_co2molemix", out_dir, title="FlowDPS Ablation: All Configs")
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
