"""Plot results from the unified posterior conditioning comparison.

Uses library functions from neural_transport.plots and neural_transport.evaluation.

Usage:
    python plot_results.py
"""

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
ALL_METHODS = ["unconditional", "dps", "flowdps", "sde", "fig", "ictm", "mcg", "pcfm", "fmps", "dflow"]


def load_multitarget_zarr(method_dir):
    import zarr

    zarr_path = method_dir / "multitarget_predictions.zarr"
    if not zarr_path.exists():
        return None

    store = zarr.open_group(str(zarr_path), mode="r")
    return {
        "predictions": np.array(store["predictions"]),
        "gt": np.array(store["gt"]),
        "obs_mask": np.array(store["obs_mask"]),
        "obs_values": np.array(store["obs_values"]),
        "target_time_idx": np.array(store["target_time_idx"]),
        "pressure_weights": np.array(store["pressure_weights"]),
        "ak": np.array(store["ak"]),
    }


def get_grid_info():
    """Get lat, lon, and pressure weights from the dataset (canonical source)."""
    from neural_transport.configs import DataConfig
    from neural_transport.data.inference_loader import GridInfo, InferenceDataLoader
    from neural_transport.inference.masking import compute_pressure_weights_from_batch

    dc = DataConfig(
        dataset="carbontracker", grid="latlon5.625", vertical_levels="l10",
        freq="6h", target_vars=["co2massmix", "p_bottom", "p_top"], forcing_vars=[],
    )
    gi = GridInfo.from_config(dc)

    # Load one sample to get p_bottom/p_top for pressure weight computation
    import os
    for base in ["/scratch/vbenson/Carbontracker", "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"]:
        if os.path.exists(os.path.join(base, "test")):
            loader = InferenceDataLoader(dc, data_path=os.path.join(base, "test"))
            dataset = loader.load_dataset()
            sample = dataset[0]
            p_bottom = sample["p_bottom"].numpy()  # [T, N, C] or [N, C]
            p_top = sample["p_top"].numpy()
            # Reshape to [nlat, nlon, nlev]
            if p_bottom.ndim == 3:
                p_bottom = p_bottom[0]  # take first timestep
                p_top = p_top[0]
            p_bottom = p_bottom.reshape(gi.nlat, gi.nlon, -1)
            p_top = p_top.reshape(gi.nlat, gi.nlon, -1)
            pw = compute_pressure_weights_from_batch(p_bottom, p_top)
            return gi.lat, gi.lon, pw
            break

    # Fallback: no dataset available, return None for pw
    logger.warning("Could not load dataset for pressure weights")
    return gi.lat, gi.lon, None


def build_osse_result(method_name, data, target_idx=0, lat=None, lon=None, pw=None, ak=None):
    """Build OSSEResult using canonical pw/ak (from dataset, not zarr)."""
    from neural_transport.inference.metrics import OSSEResult, compute_all_metrics

    preds = data["predictions"]
    gt_all = data["gt"]
    masks = data["obs_mask"]

    gt = gt_all[target_idx]
    samples = preds[target_idx]
    mask = masks[target_idx]

    metrics, extra = compute_all_metrics(samples, gt, mask_2d=mask, pressure_weights=pw, ak=ak)

    return OSSEResult(
        name=method_name, config={}, metrics=metrics,
        samples=samples, ensemble_mean=samples.mean(axis=0),
        gt=gt, mask_2d=mask, pressure_weights=pw, ak=ak,
        rank_hist=extra.get("rank_hist"),
        calibration_data=extra.get("calibration_data"),
        crps_map=extra.get("crps_map"),
        lat=lat, lon=lon,
    )


def build_all_target_metrics(data, pw=None, ak=None):
    """Compute per-target metrics using canonical pw/ak."""
    from neural_transport.inference.metrics import compute_all_metrics

    preds = data["predictions"]
    gt_all = data["gt"]
    masks = data["obs_mask"]

    per_target_metrics = []
    for t in range(preds.shape[0]):
        try:
            m, _ = compute_all_metrics(
                preds[t], gt_all[t], mask_2d=masks[t],
                pressure_weights=pw, ak=ak,
            )
            per_target_metrics.append(m)
        except Exception as e:
            logger.warning("Metrics failed for target %d: %s", t, e)
    return per_target_metrics


def plot_all():
    out_dir = EXP_DIR / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    lat, lon, canonical_pw = get_grid_info()
    # ak=ones is the default for OSSE (no real averaging kernel)
    canonical_ak = np.ones_like(canonical_pw) if canonical_pw is not None else None

    # Load data
    all_data = {}
    per_target_data = {}

    for method in ALL_METHODS:
        method_dir = EXP_DIR / "results" / method
        data = load_multitarget_zarr(method_dir)
        if data is not None:
            all_data[method] = data
            per_target_data[method] = build_all_target_metrics(data, pw=canonical_pw, ak=canonical_ak)
            logger.info("Loaded %s: %d targets x %d samples",
                        method, data["predictions"].shape[0], data["predictions"].shape[1])

    if not all_data:
        logger.error("No data found in results/!")
        return

    methods_found = list(all_data.keys())
    n_targets = next(iter(all_data.values()))["predictions"].shape[0]
    logger.info("Methods: %s (%d targets)", methods_found, n_targets)

    # Random target selection for visualization
    vis_rng = np.random.RandomState(123)
    random_targets_4 = sorted(vis_rng.choice(n_targets, size=min(4, n_targets), replace=False))
    random_targets_5 = sorted(vis_rng.choice(n_targets, size=min(5, n_targets), replace=False))

    from neural_transport.plots.conditioning_diagnostics import (
        plot_conditioning_comparison,
        plot_detail_metrics_bars,
        plot_ensemble_diagnostics,
        plot_metrics_summary,
        plot_obs_match_scatter,
        plot_per_target_panel,
        plot_power_spectra,
        plot_spread_at_unobs,
        plot_zonal_mean,
    )

    # ── Per-GT-sample conditioning comparison (5 targets) ──
    logger.info("Generating per-target conditioning comparisons (Robinson projection)...")
    for ti in random_targets_5:
        osse_for_target = {}
        for method in methods_found:
            try:
                osse_for_target[method] = build_osse_result(method, all_data[method], target_idx=ti, lat=lat, lon=lon, pw=canonical_pw, ak=canonical_ak)
            except Exception as e:
                logger.warning("Failed for %s target %d: %s", method, ti, e)

        target_dir = out_dir / f"target_{ti:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        plot_conditioning_comparison(osse_for_target, target_dir, level_idx=-1, max_samples=3)

    # ── Aggregate diagnostics ──
    osse_results = {}
    for method in methods_found:
        try:
            osse_results[method] = build_osse_result(method, all_data[method], target_idx=random_targets_5[0], lat=lat, lon=lon, pw=canonical_pw, ak=canonical_ak)
        except Exception:
            pass

    logger.info("Generating aggregate diagnostics...")
    plot_ensemble_diagnostics(osse_results, out_dir)
    plot_metrics_summary(osse_results, out_dir)
    plot_obs_match_scatter(osse_results, out_dir, level_idx=-1)
    plot_spread_at_unobs(osse_results, out_dir, level_idx=-1)
    plot_zonal_mean(osse_results, out_dir)

    # ── Per-target panels (4 random targets) ──
    logger.info("Generating per-target panels (random targets %s)...", random_targets_4)
    # Use canonical pw/ak from dataset (same for all methods)
    shared_pw, shared_ak = canonical_pw, canonical_ak

    for method in methods_found:
        data = all_data[method]
        plot_per_target_panel(
            {i: data["predictions"][t] for i, t in enumerate(random_targets_4)},
            {i: data["gt"][t] for i, t in enumerate(random_targets_4)},
            {i: data["obs_mask"][t] for i, t in enumerate(random_targets_4)},
            {i: data["obs_values"][t] for i, t in enumerate(random_targets_4)},
            out_dir, method_name=method,
            n_targets_show=4, n_samples_show=5, level_idx=-1,
            lat=lat, lon=lon,
            pressure_weights=shared_pw, ak=shared_ak,
        )

    # ── Fine-scale detail metrics (from library) ──
    logger.info("Computing fine-scale detail metrics...")
    from neural_transport.evaluation.spectral import detail_metrics, power_spectrum_2d
    from neural_transport.inference.metrics import compute_xco2_column

    dm_per_method = {}
    method_spectra = {}

    for method in methods_found:
        data = all_data[method]
        per_target_dm = []
        for t in range(n_targets):
            try:
                gt_2d = data["gt"][t].mean(axis=-1)
                pred_2d = data["predictions"][t].mean(axis=0).mean(axis=-1)
                per_target_dm.append(detail_metrics(pred_2d, gt_2d))
            except Exception:
                pass
        if per_target_dm:
            dm_per_method[method] = {
                k: float(np.mean([d[k] for d in per_target_dm]))
                for k in per_target_dm[0]
            }

        # Power spectrum for first target
        pred_2d = data["predictions"][0].mean(axis=0).mean(axis=-1)
        wn, ps = power_spectrum_2d(pred_2d)
        method_spectra[method] = (wn, ps)

    # GT spectrum
    gt_2d = next(iter(all_data.values()))["gt"][0].mean(axis=-1)
    wn_gt, ps_gt = power_spectrum_2d(gt_2d)
    method_spectra["gt"] = (wn_gt, ps_gt)

    if dm_per_method:
        methods_with_dm = [m for m in methods_found if m in dm_per_method]
        plot_detail_metrics_bars(dm_per_method, methods_with_dm, out_dir)
        plot_power_spectra(method_spectra, out_dir)

    # ── Summary table ──
    print("\n" + "=" * 140)
    print(f"Posterior Conditioning Comparison  ({len(methods_found)} methods, {n_targets} targets x 20 samples)")
    print("=" * 140)

    header = (f"{'Method':<15} {'RMSE_full':>10} {'RMSE_away':>10} {'RMSE_obs':>10} "
              f"{'SS_ratio':>10} {'Grad_ratio':>11} {'Spec_div':>10} {'HF_ratio':>10}")
    print(header)
    print("-" * len(header))

    for method in methods_found:
        ptm = per_target_data.get(method, [])
        dm = dm_per_method.get(method, {})
        if ptm:
            row = f"{method:<15}"
            row += f" {np.mean([m.rmse_3d_full for m in ptm]):>10.4f}"
            row += f" {np.mean([m.rmse_3d_away for m in ptm]):>10.4f}"
            rmse_obs_vals = [m.rmse_3d_obs for m in ptm if np.isfinite(m.rmse_3d_obs)]
            row += f" {np.mean(rmse_obs_vals):>10.4f}" if rmse_obs_vals else f" {'N/A':>10}"
            ss_vals = [m.spread_skill for m in ptm if np.isfinite(m.spread_skill)]
            row += f" {np.mean(ss_vals):>10.4f}" if ss_vals else f" {'N/A':>10}"
            row += f" {dm.get('grad_ratio', float('nan')):>11.4f}"
            row += f" {dm.get('spectral_div', float('nan')):>10.4f}"
            row += f" {dm.get('high_freq_power_ratio', float('nan')):>10.4f}"
            print(row)

    print("=" * 140 + "\n")

    # Save summary JSON
    summary = {}
    for method in methods_found:
        ptm = per_target_data.get(method, [])
        dm = dm_per_method.get(method, {})
        if ptm:
            summary[method] = {
                "mean_rmse_full": float(np.mean([m.rmse_3d_full for m in ptm])),
                "mean_rmse_away": float(np.mean([m.rmse_3d_away for m in ptm])),
                "mean_rmse_obs": float(np.nanmean([m.rmse_3d_obs for m in ptm])),
                "mean_spread_skill": float(np.nanmean([m.spread_skill for m in ptm])),
                "mean_spread": float(np.mean([m.sample_spread for m in ptm])),
                **{f"detail_{k}": v for k, v in dm.items()},
            }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("All plots saved to %s", out_dir)


if __name__ == "__main__":
    plot_all()
