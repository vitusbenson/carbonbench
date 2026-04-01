"""Publication-quality plots for posterior conditioning comparison (Phase 23).

Loads multi-target zarr data, builds OSSEResult objects, and generates:
- Per-method conditioning diagnostics (from toolkit)
- Cross-method comparison bars and Pareto front
- Per-target sample galleries
- Obs match scatter and spread at unobserved locations
- Optuna analysis

Usage:
    python plot_results.py
    python plot_results.py --exp-dir /path/to/experiment
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
METHODS = ["dps", "flowdps", "sde", "fig", "ictm"]
ALL_METHODS = ["unconditional"] + METHODS


# ── Load multi-target zarr data ──────────────────────────────────────────


def load_multitarget_zarr(method_dir):
    """Load multi-target predictions from zarr store.

    Returns dict with numpy arrays: predictions, gt, obs_mask, obs_values, etc.
    """
    import zarr

    zarr_path = method_dir / "multitarget_predictions.zarr"
    if not zarr_path.exists():
        return None

    store = zarr.open(str(zarr_path), mode="r")
    return {
        "predictions": np.array(store["predictions"]),    # [n_targets, n_samples, nlat, nlon, nlev]
        "gt": np.array(store["gt"]),                      # [n_targets, nlat, nlon, nlev]
        "obs_mask": np.array(store["obs_mask"]),           # [n_targets, nlat, nlon]
        "obs_values": np.array(store["obs_values"]),       # [n_targets, nlat, nlon]
        "target_time_idx": np.array(store["target_time_idx"]),
        "pressure_weights": np.array(store["pressure_weights"]),
        "ak": np.array(store["ak"]),
    }


def build_osse_result(method_name, data):
    """Build OSSEResult from multi-target zarr data.

    Aggregates across targets: concatenates all samples, uses first target's GT
    for comparison plots, averages mask for display.
    """
    from neural_transport.inference.metrics import OSSEResult, compute_all_metrics

    preds = data["predictions"]           # [n_targets, n_samples, nlat, nlon, nlev]
    gt_all = data["gt"]                   # [n_targets, nlat, nlon, nlev]
    masks = data["obs_mask"]              # [n_targets, nlat, nlon]
    pw = data["pressure_weights"]         # [n_targets, nlat, nlon, nlev]
    ak = data["ak"]                       # [n_targets, nlat, nlon, nlev]

    n_targets, n_samples, nlat, nlon, nlev = preds.shape

    # For metrics: compute per-target RMSE and average
    # For samples/gt: use first target for visualization
    gt_first = gt_all[0]
    samples_first = preds[0]   # [n_samples, nlat, nlon, nlev]
    mask_first = masks[0]      # [nlat, nlon]
    pw_first = pw[0] if pw.any() else None
    ak_first = ak[0] if ak.any() else None

    # Aggregate all samples across targets for overall metrics
    all_samples = preds.reshape(n_targets * n_samples, nlat, nlon, nlev)

    # Compute per-target metrics and average
    per_target_metrics = []
    per_target_extra = []
    for t in range(n_targets):
        try:
            m, extra = compute_all_metrics(
                preds[t], gt_all[t],
                mask_2d=masks[t],
                pressure_weights=pw[t] if pw.any() else None,
                ak=ak[t] if ak.any() else None,
            )
            per_target_metrics.append(m)
            per_target_extra.append(extra)
        except Exception as e:
            logger.warning("Metrics failed for target %d: %s", t, e)

    # Use first target for the OSSEResult (for visualization)
    if per_target_metrics:
        metrics = per_target_metrics[0]
        extra_first = per_target_extra[0]
    else:
        metrics, extra_first = compute_all_metrics(samples_first, gt_first, mask_2d=mask_first)

    return OSSEResult(
        name=method_name,
        config={},
        metrics=metrics,
        samples=samples_first,
        ensemble_mean=samples_first.mean(axis=0),
        gt=gt_first,
        mask_2d=mask_first,
        pressure_weights=pw_first,
        ak=ak_first,
        rank_hist=extra_first.get("rank_hist"),
        calibration_data=extra_first.get("calibration_data"),
        crps_map=extra_first.get("crps_map"),
    ), per_target_metrics


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_all(exp_dir=None):
    """Generate all publication plots."""
    if exp_dir is None:
        exp_dir = EXP_DIR
    exp_dir = Path(exp_dir)

    out_dir = exp_dir / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── A. Load multi-target data for all methods ──
    all_data = {}
    osse_results = {}
    per_target_data = {}

    for method in ALL_METHODS:
        method_dir = exp_dir / "results" / method
        data = load_multitarget_zarr(method_dir)
        if data is not None:
            all_data[method] = data
            result, per_target_m = build_osse_result(method, data)
            osse_results[method] = result
            per_target_data[method] = per_target_m
            logger.info("Loaded %s: %d targets × %d samples", method, data["predictions"].shape[0], data["predictions"].shape[1])

    if not osse_results:
        logger.warning("No multi-target zarr data found. Skipping diagnostic plots.")
    else:
        # ── B. Conditioning diagnostics (from toolkit) ──
        from neural_transport.plots.conditioning_diagnostics import (
            plot_conditioning_comparison,
            plot_ensemble_diagnostics,
            plot_metrics_summary,
            plot_obs_match_scatter,
            plot_per_target_panel,
            plot_spread_at_unobs,
            plot_xco2_maps,
            plot_zonal_mean,
        )

        logger.info("Generating conditioning comparison plots...")
        plot_conditioning_comparison(osse_results, out_dir, level_idx=-1, max_samples=3)
        plot_conditioning_comparison(osse_results, out_dir / "level0", level_idx=0, max_samples=3)
        plot_ensemble_diagnostics(osse_results, out_dir)
        plot_metrics_summary(osse_results, out_dir)
        plot_obs_match_scatter(osse_results, out_dir, level_idx=-1)
        plot_spread_at_unobs(osse_results, out_dir, level_idx=-1)

        # XCO2 maps (only if pressure weights available)
        has_pw = any(r.pressure_weights is not None for r in osse_results.values())
        if has_pw:
            plot_xco2_maps(osse_results, out_dir)

        plot_zonal_mean(osse_results, out_dir)

        # ── C. Per-target sample galleries (for best methods) ──
        for method in ALL_METHODS:
            if method not in all_data:
                continue
            data = all_data[method]
            n_targets = data["predictions"].shape[0]
            samples_dict = {t: data["predictions"][t] for t in range(n_targets)}
            gt_dict = {t: data["gt"][t] for t in range(n_targets)}
            mask_dict = {t: data["obs_mask"][t] for t in range(n_targets)}
            obs_dict = {t: data["obs_values"][t] for t in range(n_targets)}

            plot_per_target_panel(
                samples_dict, gt_dict, mask_dict, obs_dict,
                out_dir, method_name=method,
                n_targets_show=4, n_samples_show=5, level_idx=-1,
            )

        # ── D. Print summary table ──
        _print_summary(osse_results, per_target_data, exp_dir)

    # ── E. Optuna analysis ──
    _plot_optuna(exp_dir, out_dir)

    # ── F. Cross-method bars (from method_info.json if available) ──
    _plot_cross_method_bars(exp_dir, out_dir)

    logger.info("All plots saved to %s", out_dir)


def _print_summary(osse_results, per_target_data, exp_dir):
    """Print summary table with per-method metrics."""
    print("\n" + "=" * 90)
    print("GRAND SUMMARY — Multi-Target Posterior Conditioning Comparison")
    print("=" * 90)

    header = f"{'Method':<15} {'RMSE_3D':>10} {'RMSE_obs':>10} {'RMSE_away':>10} {'R2':>8} {'SS_ratio':>10} {'Spread':>10}"
    print(header)
    print("-" * len(header))

    for method in ALL_METHODS:
        if method not in osse_results:
            continue
        res = osse_results[method]
        m = res.metrics
        row = f"{method:<15}"
        row += f" {m.rmse_3d_full:>10.4f}"
        row += f" {m.rmse_3d_obs:>10.4f}"
        row += f" {m.rmse_3d_away:>10.4f}"
        row += f" {m.r2:>8.4f}"
        row += f" {m.spread_skill:>10.4f}"
        row += f" {m.sample_spread:>10.4f}"
        print(row)

    # Success gate
    uncond = osse_results.get("unconditional")
    if uncond is not None:
        best_method = None
        best_rmse = float("inf")
        for method in METHODS:
            if method in osse_results:
                rmse = osse_results[method].metrics.rmse_3d_full
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_method = method

        print(f"\nUnconditional RMSE: {uncond.metrics.rmse_3d_full:.4f}")
        if best_method:
            improvement = 1 - best_rmse / uncond.metrics.rmse_3d_full
            print(f"Best method: {best_method} (RMSE={best_rmse:.4f}, {improvement*100:.1f}% improvement)")
    print("=" * 90 + "\n")


def _plot_optuna(exp_dir, out_dir):
    """Run Optuna analysis for each method."""
    try:
        from neural_transport.training.study_analysis import analyze_study
    except ImportError:
        return

    for method in METHODS:
        db_path = exp_dir / "optuna_runs" / f"{method}_study.db"
        if not db_path.exists():
            continue
        try:
            analyze_study(
                f"sqlite:///{db_path}",
                study_name=f"posterior_{method}",
                out_dir=out_dir / f"optuna_{method}",
                run_dir=exp_dir / "optuna_runs" / method,
            )
        except Exception as e:
            logger.warning("Optuna analysis failed for %s: %s", method, e)


def _plot_cross_method_bars(exp_dir, out_dir):
    """Summary bars from method_info.json files."""
    from neural_transport.plots.metrics_plots import plot_pareto_front, plot_summary_bars

    results_dir = exp_dir / "results"
    metrics = {}
    for method in ALL_METHODS:
        info_path = results_dir / method / "method_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            metrics[method] = {"wall_time_sec": info.get("wall_time_sec", 0)}

    if metrics and any("wall_time_sec" in m for m in metrics.values()):
        # Need RMSE from osse_results — load if available
        for method in ALL_METHODS:
            data = load_multitarget_zarr(results_dir / method)
            if data is not None and method in metrics:
                from neural_transport.inference.metrics import compute_all_metrics
                preds = data["predictions"]
                gt = data["gt"]
                # Use first target for quick RMSE
                m, _ = compute_all_metrics(preds[0], gt[0])
                metrics[method]["RMSE_3D"] = m.rmse_3d_full
                metrics[method]["R2"] = m.r2

        if any("RMSE_3D" in m for m in metrics.values()):
            plot_summary_bars(metrics, "RMSE_3D", out_dir, title="3D RMSE by Method")
            plot_summary_bars(metrics, "R2", out_dir, title="R² by Method", lower_is_better=False)
            plot_pareto_front(metrics, "wall_time_sec", "RMSE_3D", out_dir, title="Cost vs Accuracy")


def main():
    parser = argparse.ArgumentParser(description="Plot multi-target comparison results")
    parser.add_argument("--exp-dir", type=str, default=None)
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir) if args.exp_dir else EXP_DIR
    plot_all(exp_dir)


if __name__ == "__main__":
    main()
