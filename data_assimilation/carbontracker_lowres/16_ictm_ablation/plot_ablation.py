"""Plot ICTM ablation results (Phase 9).

Custom plots: r_max sweep, n_inner_steps sweep, r_schedule, inner_lr, ICTM vs baselines.
"""

import argparse
from pathlib import Path

from neural_transport.plots.metrics_plots import load_ablation_results, plot_ablation_sweep, plot_summary_bars


def plot_rmse_vs_rmax(results, out_dir):
    r_configs = {k: v for k, v in results.items() if k.startswith("ictm_rmax")}
    if r_configs:
        plot_ablation_sweep(
            r_configs, "RMSE_3D_co2molemix",
            lambda name: float(name[len("ictm_rmax"):]),
            out_dir, xlabel="r_max", log_x=True,
            title="ICTM: RMSE and Spread vs r_max",
            second_metric_key="RelRMSE_3D_co2molemix", second_ylabel="RelRMSE",
        )


def plot_rmse_vs_inner_steps(results, out_dir):
    n_configs = {k: v for k, v in results.items() if k.startswith("ictm_inner")}
    if n_configs:
        plot_ablation_sweep(
            n_configs, "RMSE_3D_co2molemix",
            lambda name: int(name[len("ictm_inner"):]),
            out_dir, xlabel="n_inner_steps",
            title="ICTM: RMSE and Spread vs n_inner_steps",
            second_metric_key="RelRMSE_3D_co2molemix", second_ylabel="RelRMSE",
        )


def plot_r_schedule(results, out_dir):
    sched_configs = {k: v for k, v in results.items() if k.startswith("ictm_sched_")}
    if sched_configs:
        plot_summary_bars(sched_configs, "RMSE_3D_co2molemix", out_dir,
                          title="ICTM: r_schedule Comparison")


def plot_inner_lr(results, out_dir):
    lr_configs = {k: v for k, v in results.items() if k.startswith("ictm_lr")}
    if lr_configs:
        plot_summary_bars(lr_configs, "RMSE_3D_co2molemix", out_dir,
                          title="ICTM: inner_lr Comparison (n_inner_steps=5)")


def plot_ictm_vs_baselines(results, out_dir):
    compare = {k: v for k, v in results.items()
               if k in ("unconditional", "best_flowdps", "best_sde", "best_fig", "ictm_rmax1.0")}
    if compare:
        plot_summary_bars(compare, "RMSE_lat_co2molemix", out_dir,
                          title="ICTM vs All Baselines")


CUSTOM_PLOTS = [plot_rmse_vs_rmax, plot_rmse_vs_inner_steps,
                plot_r_schedule, plot_inner_lr, plot_ictm_vs_baselines]


def main():
    parser = argparse.ArgumentParser(description="Plot ICTM ablation results")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parent / "results"
    results = load_ablation_results(results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for plot_fn in CUSTOM_PLOTS:
        plot_fn(results, out_dir)
    plot_summary_bars(results, "RMSE_lat_co2molemix", out_dir, title="ICTM Ablation: All Configs")
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
