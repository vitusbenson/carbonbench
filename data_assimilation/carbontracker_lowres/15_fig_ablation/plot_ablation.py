"""Plot FIG ablation results (Phase 8).

Custom plots: step_size_c sweep, k_steps sweep, noise_scale_w, FIG vs baselines.
"""

import argparse
from pathlib import Path

from neural_transport.plots.metrics_plots import load_ablation_results, plot_ablation_sweep, plot_summary_bars


def plot_rmse_vs_step_size(results, out_dir):
    c_configs = {k: v for k, v in results.items()
                 if k.startswith("fig_c") and k[5:].replace(".", "").isdigit()}
    if c_configs:
        plot_ablation_sweep(
            c_configs, "RMSE_3D_co2molemix",
            lambda name: float(name[5:]),
            out_dir, xlabel="step_size_c", log_x=True,
            title="FIG: RMSE and Spread vs step_size_c",
            second_metric_key="RelRMSE_3D_co2molemix", second_ylabel="RelRMSE",
        )


def plot_rmse_vs_k_steps(results, out_dir):
    k_configs = {k: v for k, v in results.items() if k.startswith("fig_k")}
    if k_configs:
        plot_ablation_sweep(
            k_configs, "RMSE_3D_co2molemix",
            lambda name: int(name[5:]),
            out_dir, xlabel="k_steps",
            title="FIG: RMSE and Spread vs k_steps",
            second_metric_key="RelRMSE_3D_co2molemix", second_ylabel="RelRMSE",
        )


def plot_noise_scale(results, out_dir):
    w_configs = {k: v for k, v in results.items() if k.startswith("fig_w")}
    if w_configs:
        plot_summary_bars(w_configs, "RMSE_3D_co2molemix", out_dir,
                          title="FIG: Noise Scale w Comparison")


def plot_fig_vs_baselines(results, out_dir):
    compare = {k: v for k, v in results.items()
               if k in ("unconditional", "best_flowdps", "best_sde", "fig_c10")}
    if compare:
        plot_summary_bars(compare, "RMSE_lat_co2molemix", out_dir,
                          title="FIG vs FlowDPS vs SDE Comparison")


CUSTOM_PLOTS = [plot_rmse_vs_step_size, plot_rmse_vs_k_steps,
                plot_noise_scale, plot_fig_vs_baselines]


def main():
    parser = argparse.ArgumentParser(description="Plot FIG ablation results")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parent / "results"
    results = load_ablation_results(results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for plot_fn in CUSTOM_PLOTS:
        plot_fn(results, out_dir)
    plot_summary_bars(results, "RMSE_lat_co2molemix", out_dir, title="FIG Ablation: All Configs")
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
