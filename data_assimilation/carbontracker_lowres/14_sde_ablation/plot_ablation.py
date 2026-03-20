"""Plot SDE posterior sampling ablation results (Phase 7).

Custom plots: sigma_max sweep (dual-axis), noise schedule, corrector sweep, ODE vs SDE.
"""

import argparse
from pathlib import Path

from neural_transport.plots.metrics_plots import load_ablation_results, plot_ablation_sweep, plot_summary_bars


def plot_rmse_spread_vs_sigma_max(results, out_dir):
    smax_configs = {k: v for k, v in results.items()
                    if k.startswith("sde_smax") and "_" not in k.split("smax")[1]}
    if smax_configs:
        plot_ablation_sweep(
            smax_configs, "RMSE_3D_co2molemix",
            lambda name: float(name.split("smax")[1]),
            out_dir, xlabel="sigma_max", log_x=True,
            title="SDE: RMSE and Spread vs sigma_max",
            second_metric_key="RelRMSE_3D_co2molemix", second_ylabel="RelRMSE",
        )


def plot_noise_schedule(results, out_dir):
    sched_configs = {k: v for k, v in results.items() if k.startswith("sde_sched_")}
    if sched_configs:
        plot_summary_bars(sched_configs, "RMSE_3D_co2molemix", out_dir,
                          title="SDE: Noise Schedule Comparison")


def plot_corrector_sweep(results, out_dir):
    pc_configs = {k: v for k, v in results.items()
                  if k.startswith("pc_c") and "eps" not in k}
    if "sde_smax0.3" in results:
        pc_configs["no_corrector"] = results["sde_smax0.3"]
    if pc_configs:
        plot_summary_bars(pc_configs, "RMSE_3D_co2molemix", out_dir,
                          title="SDE: Corrector Steps Sweep")


def plot_ode_vs_sde(results, out_dir):
    compare = {k: v for k, v in results.items()
               if k in ("unconditional", "best_flowdps", "sde_smax0.3")}
    if compare:
        plot_summary_bars(compare, "RMSE_lat_co2molemix", out_dir,
                          title="ODE (FlowDPS) vs SDE Comparison")


CUSTOM_PLOTS = [plot_rmse_spread_vs_sigma_max, plot_noise_schedule,
                plot_corrector_sweep, plot_ode_vs_sde]


def main():
    parser = argparse.ArgumentParser(description="Plot SDE ablation results")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).resolve().parent / "results"
    results = load_ablation_results(results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for plot_fn in CUSTOM_PLOTS:
        plot_fn(results, out_dir)
    plot_summary_bars(results, "RMSE_lat_co2molemix", out_dir, title="SDE Ablation: All Configs")
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
