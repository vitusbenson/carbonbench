"""Phase 25 plots + metrics aggregation.

Produces:
- Error-growth curves (rmse, spread, crps, spread/err) per method overlaid.
- Per-target panels for the 1-day eval (GT | Obs | Samples | Ens.Mean | Error).
- A trajectory animation showing GT vs posterior-conditioned ensemble mean
  for one selected init.
- Metrics summary table comparing Phase 23f (instantaneous, no transport prior)
  vs Phase 25 (with transport prior) for 1-day eval.

Usage:
    python plot_results.py                   # use whatever is in results/
    python plot_results.py --tag 1month
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
PHASE23_RESULTS = EXP_DIR.parent / "13_posterior_conditioning_ablation" / "results"


def _to_xco2(field_3d, pw, ak):
    if pw is None or ak is None:
        return field_3d.mean(axis=-1)
    weighted = pw * ak * field_3d
    return weighted.sum(axis=-1)


def plot_error_growth(per_lead_csvs: dict[str, Path], out_path: Path, freq_hours=6):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    metrics = ["rmse_mean", "spread", "crps", "spread_error_ratio"]
    for ax, m in zip(axes.flatten(), metrics):
        for label, csv in per_lead_csvs.items():
            if not csv.exists():
                continue
            df = pd.read_csv(csv).set_index("lead")
            t_days = df.index.values * freq_hours / 24.0
            ax.plot(t_days, df[m], label=label, lw=1.8)
        ax.set_title(m)
        ax.set_xlabel("lead time [days]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Error growth → %s", out_path)


def plot_per_target_panels_1day(method_results: dict[str, Path], out_dir: Path,
                                 grid_info: dict, n_targets_show=4, n_samples_show=5):
    """For each method with a 1-day multitarget zarr, build per-target panels."""
    from neural_transport.plots.conditioning_diagnostics import plot_per_target_panel
    from neural_transport.datasets.grids import LATLON_PROTOTYPE_COORDS

    lat = LATLON_PROTOTYPE_COORDS["latlon5.625"]["lat"]
    lon = LATLON_PROTOTYPE_COORDS["latlon5.625"]["lon"]

    for method, result_dir in method_results.items():
        zpath = result_dir / "multitarget_predictions.zarr"
        if not zpath.exists():
            logger.info("[%s] no 1-day zarr, skipping panel", method)
            continue
        store = zarr.open(str(zpath), mode="r")
        preds = store["predictions"][:]  # [T, S, nlat, nlon, C]
        gt = store["gt"][:]  # [T, nlat, nlon, C]
        obs_mask = store["obs_mask"][:]
        obs_values = store["obs_values"][:]
        pw = store["pressure_weights"][:] if "pressure_weights" in store else None
        ak = store["ak"][:] if "ak" in store else None

        samples_per = {i: preds[i] for i in range(preds.shape[0])}
        gt_per = {i: gt[i] for i in range(gt.shape[0])}
        mask_per = {i: obs_mask[i] for i in range(obs_mask.shape[0])}
        obs_per = {i: obs_values[i] for i in range(obs_values.shape[0])}
        pw_first = pw[0] if pw is not None else None
        ak_first = ak[0] if ak is not None else None

        panel_dir = out_dir / f"panels_{method}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        plot_per_target_panel(
            samples_per, gt_per, mask_per, obs_per,
            out_dir=panel_dir,
            method_name=method,
            n_targets_show=n_targets_show,
            n_samples_show=n_samples_show,
            level_idx=-1,
            lat=lat, lon=lon,
            pressure_weights=pw_first, ak=ak_first,
            imgformats=["png"],
        )
        logger.info("[%s] panels → %s", method, panel_dir)


def plot_trajectory_animation(traj_dir: Path, out_path: Path, level_idx=-1,
                              sample_idx=0, init_idx=0, max_frames=120, stride=1):
    import matplotlib.animation as animation

    preds = xr.open_zarr(str(traj_dir / "preds_trajectory.zarr"))
    gt = xr.open_zarr(str(traj_dir / "gt_trajectory.zarr"))
    target = "co2massmix"

    pr = preds[target].isel(init=init_idx).mean("sample").values  # [L, nlat, nlon, C]
    gtv = gt[target].isel(init=init_idx).values  # [L, nlat, nlon, C]
    if level_idx == -1:
        pr2d = pr.mean(axis=-1)
        gt2d = gtv.mean(axis=-1)
    else:
        pr2d = pr[..., level_idx]
        gt2d = gtv[..., level_idx]
    err2d = pr2d - gt2d

    L = min(pr2d.shape[0], max_frames * stride)
    frame_idxs = list(range(0, L, stride))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin = float(np.nanpercentile(gt2d[:L], 2))
    vmax = float(np.nanpercentile(gt2d[:L], 98))
    emax = float(np.nanpercentile(np.abs(err2d[:L]), 98))
    im_gt = axes[0].imshow(gt2d[0], vmin=vmin, vmax=vmax, cmap="Spectral_r", origin="lower")
    im_pr = axes[1].imshow(pr2d[0], vmin=vmin, vmax=vmax, cmap="Spectral_r", origin="lower")
    im_er = axes[2].imshow(err2d[0], vmin=-emax, vmax=emax, cmap="RdBu_r", origin="lower")
    fig.colorbar(im_gt, ax=axes[0], fraction=0.03)
    fig.colorbar(im_pr, ax=axes[1], fraction=0.03)
    fig.colorbar(im_er, ax=axes[2], fraction=0.03)
    axes[0].set_title("GT")
    axes[1].set_title("Ens mean")
    axes[2].set_title("Error")
    title = fig.suptitle("lead 0")

    def update(fi):
        i = frame_idxs[fi]
        im_gt.set_data(gt2d[i])
        im_pr.set_data(pr2d[i])
        im_er.set_data(err2d[i])
        title.set_text(f"lead {i}")
        return im_gt, im_pr, im_er, title

    ani = animation.FuncAnimation(fig, update, frames=len(frame_idxs), interval=120, blit=False)
    try:
        ani.save(str(out_path), writer="pillow", dpi=90)
    except Exception as e:
        logger.warning("Pillow writer failed (%s); trying ffmpeg", e)
        ani.save(str(out_path).replace(".gif", ".mp4"), writer="ffmpeg", dpi=90)
    plt.close(fig)
    logger.info("Animation → %s", out_path)


def build_phase23_comparison(out_path: Path):
    """Load Phase 23f + Phase 25 1-day results, emit side-by-side metrics."""
    rows = []
    specs = [
        (23, PHASE23_RESULTS, "unconditional", "unconditional"),
        (23, PHASE23_RESULTS, "fmps", "fmps"),
        (23, PHASE23_RESULTS, "dflow", "dflow"),
        (25, EXP_DIR / "results", "unconditional", "unconditional_1day"),
        (25, EXP_DIR / "results", "fmps", "fmps_1day"),
        (25, EXP_DIR / "results", "dflow", "dflow_1day"),
    ]
    for phase, results_root, method, subdir in specs:
            method_dir = results_root / subdir
            zpath = method_dir / "multitarget_predictions.zarr"
            if not zpath.exists():
                logger.info("[phase%d/%s] missing %s", phase, method, zpath)
                continue
            store = zarr.open(str(zpath), mode="r")
            preds = store["predictions"][:]   # [T, S, nlat, nlon, C]
            gt = store["gt"][:]               # [T, nlat, nlon, C]
            ensemble_mean = preds.mean(axis=1)
            rmse = float(np.sqrt(((ensemble_mean - gt) ** 2).mean()))
            spread = float(preds.std(axis=1).mean())
            rmse_full = float(np.sqrt(((preds - gt[:, None]) ** 2).mean()))
            rows.append({
                "phase": phase,
                "method": method,
                "rmse_mean": rmse,
                "rmse_full": rmse_full,
                "spread": spread,
                "spread_error_ratio": spread / max(rmse, 1e-12),
                "n_targets": preds.shape[0],
                "n_samples": preds.shape[1],
                "zarr": str(zpath),
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        logger.info("Phase-23 vs Phase-25 comparison → %s\n%s", out_path, df.to_string(index=False))
    else:
        logger.warning("No comparison rows produced.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None, help="trajectory tag, e.g. 1month")
    p.add_argument("--freq-hours", type=int, default=6)
    args = p.parse_args()

    plots_dir = EXP_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1) error-growth curves per trajectory eval tag
    results_root = EXP_DIR / "results"
    if args.tag is not None:
        tags = [args.tag]
    else:
        tags = sorted({d.name.split("_", 1)[1] for d in results_root.glob("*_*")
                        if (d / "scores" / "metrics_per_lead.csv").exists()})
    for tag in tags:
        per_lead_csvs = {}
        for method in ["none", "fmps", "dflow"]:
            csv = results_root / f"{method}_{tag}" / "scores" / "metrics_per_lead.csv"
            if csv.exists():
                per_lead_csvs[method] = csv
        if per_lead_csvs:
            plot_error_growth(per_lead_csvs,
                              plots_dir / f"error_growth_{tag}.png",
                              freq_hours=args.freq_hours)

    # 2) per-target panels for the 1-day eval
    method_1day = {m: results_root / f"{m}_1day" for m in ["fmps", "dflow", "unconditional"]
                   if (results_root / f"{m}_1day" / "multitarget_predictions.zarr").exists()}
    if method_1day:
        plot_per_target_panels_1day(method_1day, plots_dir, grid_info={})

    # 3) animation for the longest trajectory eval
    for tag in tags:
        for method in ["fmps", "dflow", "none"]:
            traj_dir = results_root / f"{method}_{tag}"
            if (traj_dir / "preds_trajectory.zarr").exists():
                ani_path = plots_dir / f"animation_{method}_{tag}.gif"
                try:
                    plot_trajectory_animation(traj_dir, ani_path)
                except Exception as e:
                    logger.warning("Animation failed for %s/%s: %s", method, tag, e)

    # 4) Phase 23 vs Phase 25 comparison table
    build_phase23_comparison(plots_dir / "compare_phase23_phase25.csv")


if __name__ == "__main__":
    main()
