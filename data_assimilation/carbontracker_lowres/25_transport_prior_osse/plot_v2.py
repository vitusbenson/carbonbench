"""Phase 25b polished plots: Robinson-projection MP4 animations, multi-method
comparison panels at +6h / +7d / +30d, individual-trajectory panels, and
publication-style metric curves.

Inputs are read from `results/{method}_{tag}/preds_trajectory.zarr` and
`gt_trajectory.zarr` (for trajectory tags like 1month / full) and from
`results/{method}_1day/multitarget_predictions.zarr` for the 1-day eval.

Usage:
    python plot_v2.py                       # everything
    python plot_v2.py --tag 1month          # only the 1-month figures
    python plot_v2.py --skip-animations     # static plots only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import shutil as _shutil
import sys as _sys

# Help matplotlib find ffmpeg shipped inside the conda env.
_ffmpeg = _shutil.which("ffmpeg") or str(Path(_sys.executable).parent / "ffmpeg")
if Path(_ffmpeg).exists():
    matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
PLOTS = EXP_DIR / "plots" / "v2"
PLOTS.mkdir(parents=True, exist_ok=True)

METHODS_ORDER = ["none", "fmps", "dflow", "dflow_window"]
METHOD_LABEL = {
    "none": "Unconditional AR",
    "fmps": "FMPS",
    "dflow": "D-Flow (per-step)",
    "dflow_window": "D-Flow (window)",
    "unconditional": "Unconditional",
}
COLOR = {
    "none": "#888888",
    "fmps": "#1f77b4",
    "dflow": "#d62728",
    "dflow_window": "#9467bd",
    "unconditional": "#888888",
}

# CO2 unit conversion: dataset stores mass-mixing-ratio (mg/kg). Convert to
# volume-mixing-ratio (ppm) by molar mass ratio: ppm = mmr * (M_air/M_co2)
PPM_FACTOR = 28.97 / 44.01


def _column_avg(field, pw=None):
    """Pressure-weighted column mean over the level axis (last dim)."""
    if pw is None:
        return field.mean(axis=-1)
    pw = pw / pw.sum(axis=-1, keepdims=True)
    return (field * pw).sum(axis=-1)


def _to_ppm(x):
    return x * PPM_FACTOR


def _date_str(t):
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def _open_traj(method, tag):
    p = RESULTS / f"{method}_{tag}"
    pf = p / "preds_trajectory.zarr"
    gf = p / "gt_trajectory.zarr"
    if not pf.exists() or not gf.exists():
        return None, None
    return xr.open_zarr(str(pf)), xr.open_zarr(str(gf))


# --------------------------------------------------------------------------- #
# Robinson projection helpers
# --------------------------------------------------------------------------- #

def _new_robinson_axis(fig, gridspec_pos, title=None):
    ax = fig.add_subplot(gridspec_pos, projection=ccrs.Robinson(central_longitude=0))
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def _imshow_robinson(ax, data2d, lon, lat, vmin, vmax, cmap):
    # Roll lon to center 0
    if lon[0] >= 0:
        roll = np.searchsorted(lon, 180.0)
        data2d = np.roll(data2d, -roll, axis=1)
        lon_plot = np.where(lon >= 180, lon - 360, lon)
        lon_plot = np.roll(lon_plot, -roll)
    else:
        lon_plot = lon
    return ax.pcolormesh(
        lon_plot, lat, data2d,
        transform=ccrs.PlateCarree(),
        vmin=vmin, vmax=vmax, cmap=cmap, shading="auto",
    )


# --------------------------------------------------------------------------- #
# Animations
# --------------------------------------------------------------------------- #

def animate_method(method, tag, *, individual_sample=None, individual_init=None,
                   max_frames=180, fps=12, sample_stride=None):
    """Robinson-projection animation: GT | Pred | Error.

    If individual_sample is None: plot ensemble mean over samples.
    Else: plot the chosen sample trajectory.
    """
    preds, gt = _open_traj(method, tag)
    if preds is None:
        logger.info("[%s/%s] no trajectory zarr, skip", method, tag)
        return

    target = "co2massmix"
    init_idx = individual_init if individual_init is not None else 0
    pr = preds[target].isel(init=init_idx).values  # [S, L, lat, lon, lev]
    gtv = gt[target].isel(init=init_idx).values  # [L, lat, lon, lev]
    times = preds["time"].isel(init=init_idx).values  # [L]

    # column-average over levels
    pr_col = _to_ppm(pr.mean(axis=-1))     # [S, L, lat, lon]
    gt_col = _to_ppm(gtv.mean(axis=-1))    # [L, lat, lon]

    if individual_sample is None:
        pr2d = pr_col.mean(axis=0)         # [L, lat, lon]
        title_tag = "ensemble mean"
        suffix = "ensmean"
    else:
        pr2d = pr_col[individual_sample]   # [L, lat, lon]
        title_tag = f"sample {individual_sample}"
        suffix = f"traj_s{individual_sample}_init{init_idx}"
    err2d = pr2d - gt_col

    L = min(pr2d.shape[0], max_frames)
    if sample_stride is None:
        sample_stride = max(1, pr2d.shape[0] // max_frames)
    frame_idxs = list(range(0, pr2d.shape[0], sample_stride))[:max_frames]

    lat = preds["lat"].values
    lon = preds["lon"].values

    vmin = float(np.nanpercentile(gt_col[:L], 2))
    vmax = float(np.nanpercentile(gt_col[:L], 98))
    emax = float(np.nanpercentile(np.abs(err2d[:L]), 98))
    if emax < 1e-6:
        emax = 1.0

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.05)
    ax_gt = _new_robinson_axis(fig, gs[0, 0], "GT")
    ax_pr = _new_robinson_axis(fig, gs[0, 1], f"{METHOD_LABEL.get(method, method)} ({title_tag})")
    ax_er = _new_robinson_axis(fig, gs[0, 2], "Error (Pred − GT)")

    im_gt = _imshow_robinson(ax_gt, gt_col[0], lon, lat, vmin, vmax, "Spectral_r")
    im_pr = _imshow_robinson(ax_pr, pr2d[0], lon, lat, vmin, vmax, "Spectral_r")
    im_er = _imshow_robinson(ax_er, err2d[0], lon, lat, -emax, emax, "RdBu_r")

    cb1 = fig.colorbar(im_gt, ax=[ax_gt, ax_pr], orientation="horizontal",
                       shrink=0.6, pad=0.04, aspect=40)
    cb1.set_label("Column-mean CO₂ [ppm]")
    cb2 = fig.colorbar(im_er, ax=ax_er, orientation="horizontal",
                       shrink=0.7, pad=0.04, aspect=20)
    cb2.set_label("Error [ppm]")

    suptitle = fig.suptitle("", fontsize=12, y=0.97)

    def update(fi):
        i = frame_idxs[fi]
        # Roll-and-update each
        for ax, im, data in [
            (ax_gt, im_gt, gt_col[i]),
            (ax_pr, im_pr, pr2d[i]),
            (ax_er, im_er, err2d[i]),
        ]:
            if lon[0] >= 0:
                roll = np.searchsorted(lon, 180.0)
                data = np.roll(data, -roll, axis=1)
            im.set_array(data.ravel())
        date = _date_str(times[i])
        days = i * 6 / 24.0
        suptitle.set_text(f"{METHOD_LABEL.get(method, method)} — {tag} — {date}  (lead +{days:.2f} d)")
        return im_gt, im_pr, im_er, suptitle

    ani = animation.FuncAnimation(fig, update, frames=len(frame_idxs),
                                  interval=1000 / fps, blit=False)
    out = PLOTS / f"animation_{method}_{tag}_{suffix}.mp4"
    try:
        writer = animation.FFMpegWriter(fps=fps, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p"])
        ani.save(str(out), writer=writer, dpi=110)
        logger.info("Animation → %s", out)
    except Exception as e:
        gif = out.with_suffix(".gif")
        logger.warning("ffmpeg failed (%s); fallback to %s", e, gif)
        ani.save(str(gif), writer="pillow", dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Method-comparison panel at fixed lead times
# --------------------------------------------------------------------------- #

def comparison_at_leads(tag, leads, *, n_inits_show=5, seed=0):
    """For each chosen lead, side-by-side ensemble-mean column XCO2 of all
    available methods + GT, for n_inits_show random inits.
    """
    methods = []
    data = {}
    for m in METHODS_ORDER:
        preds, gt = _open_traj(m, tag)
        if preds is None:
            continue
        methods.append(m)
        data[m] = (preds, gt)
    if not methods:
        logger.info("[%s] no methods with trajectory data", tag)
        return

    base_preds, base_gt = data[methods[0]]
    n_inits = base_preds.sizes["init"]
    rng = np.random.RandomState(seed)
    init_pick = sorted(rng.choice(n_inits, min(n_inits_show, n_inits), replace=False).tolist())
    times = base_preds["time"].values  # [init, lead]
    lat = base_preds["lat"].values
    lon = base_preds["lon"].values

    for lead in leads:
        if lead >= base_preds.sizes["lead"]:
            logger.info("lead %d not in tag %s (max=%d), skip", lead, tag, base_preds.sizes["lead"])
            continue
        days = lead * 6 / 24.0

        ncols = 1 + len(methods)
        nrows = len(init_pick)
        fig = plt.figure(figsize=(3.0 * ncols, 2.0 * nrows + 0.6))
        gs = fig.add_gridspec(nrows, ncols, wspace=0.05, hspace=0.15)

        # Determine shared colormap range from the GT at this lead.
        gt_stack = np.stack(
            [_to_ppm(data[methods[0]][1]["co2massmix"].isel(init=i, lead=lead).values.mean(axis=-1))
             for i in init_pick],
            axis=0,
        )
        vmin = float(np.nanpercentile(gt_stack, 2))
        vmax = float(np.nanpercentile(gt_stack, 98))

        for r, init_idx in enumerate(init_pick):
            gtv = _to_ppm(
                data[methods[0]][1]["co2massmix"].isel(init=init_idx, lead=lead).values.mean(axis=-1)
            )
            ax = _new_robinson_axis(fig, gs[r, 0],
                                    f"GT  (init {int(base_preds.init.values[init_idx])})" if r == 0 else None)
            _imshow_robinson(ax, gtv, lon, lat, vmin, vmax, "Spectral_r")
            ax.set_ylabel(f"{_date_str(times[init_idx, lead])}", fontsize=8)

            for c, m in enumerate(methods, start=1):
                preds, _ = data[m]
                pred_em = _to_ppm(
                    preds["co2massmix"].isel(init=init_idx, lead=lead)
                    .mean(dim="sample").values.mean(axis=-1)
                )
                title = METHOD_LABEL.get(m, m) if r == 0 else None
                ax2 = _new_robinson_axis(fig, gs[r, c], title)
                _imshow_robinson(ax2, pred_em, lon, lat, vmin, vmax, "Spectral_r")

        fig.suptitle(f"Method comparison @ lead +{days:.1f} d ({tag})", y=0.995, fontsize=12)
        out = PLOTS / f"compare_methods_lead{lead:04d}_{tag}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("Comparison panel → %s", out)


# --------------------------------------------------------------------------- #
# Per-method trajectory evolution panel (individual samples)
# --------------------------------------------------------------------------- #

def trajectory_evolution_panel(method, tag, *, leads=None, n_samples_show=5,
                               init_idx=0):
    preds, gt = _open_traj(method, tag)
    if preds is None:
        return
    if leads is None:
        L = preds.sizes["lead"]
        leads = [1, max(1, L // 8), max(2, L // 2), L - 1]
    leads = [l for l in leads if l < preds.sizes["lead"]]
    n_samples = min(n_samples_show, preds.sizes["sample"])
    times = preds["time"].isel(init=init_idx).values
    lat = preds["lat"].values
    lon = preds["lon"].values
    target = "co2massmix"

    nrows = 1 + n_samples  # GT row + sample rows
    ncols = len(leads)
    fig = plt.figure(figsize=(3.0 * ncols, 2.0 * nrows + 0.5))
    gs = fig.add_gridspec(nrows, ncols, wspace=0.05, hspace=0.15)

    gts = np.stack([_to_ppm(gt[target].isel(init=init_idx, lead=l).values.mean(axis=-1))
                    for l in leads], axis=0)
    vmin = float(np.nanpercentile(gts, 2))
    vmax = float(np.nanpercentile(gts, 98))

    for c, lead in enumerate(leads):
        days = lead * 6 / 24.0
        ax = _new_robinson_axis(fig, gs[0, c],
                                f"+{days:.1f} d   {_date_str(times[lead])}")
        _imshow_robinson(ax, gts[c], lon, lat, vmin, vmax, "Spectral_r")
        if c == 0:
            ax.text(-0.1, 0.5, "GT", transform=ax.transAxes, fontsize=11,
                    ha="right", va="center", rotation=90)

        for s in range(n_samples):
            pr = _to_ppm(
                preds[target].isel(init=init_idx, sample=s, lead=lead)
                .values.mean(axis=-1)
            )
            ax2 = _new_robinson_axis(fig, gs[s + 1, c], None)
            _imshow_robinson(ax2, pr, lon, lat, vmin, vmax, "Spectral_r")
            if c == 0:
                ax2.text(-0.1, 0.5, f"sample {s}", transform=ax2.transAxes,
                         fontsize=10, ha="right", va="center", rotation=90)

    fig.suptitle(f"{METHOD_LABEL.get(method, method)} — individual trajectories ({tag})",
                 y=0.995, fontsize=12)
    out = PLOTS / f"traj_evolution_{method}_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trajectory-evolution → %s", out)


# --------------------------------------------------------------------------- #
# Polished metrics plot
# --------------------------------------------------------------------------- #

def metrics_plot(tag, freq_hours=6):
    rows = {}
    for m in METHODS_ORDER:
        csv = RESULTS / f"{m}_{tag}" / "scores" / "metrics_per_lead.csv"
        if csv.exists():
            rows[m] = pd.read_csv(csv).set_index("lead")
    if not rows:
        logger.info("[%s] no per-lead csvs", tag)
        return
    metrics = [
        ("rmse_mean", "RMSE [ppm·M_air/M_co2]"),
        ("crps", "CRPS"),
        ("spread", "Ensemble spread"),
        ("spread_error_ratio", "spread / RMSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (col, ylabel) in zip(axes.flatten(), metrics):
        for m, df in rows.items():
            t_days = df.index.values * freq_hours / 24.0
            if col not in df.columns:
                continue
            ax.plot(t_days, df[col], label=METHOD_LABEL.get(m, m), lw=1.8,
                    color=COLOR.get(m))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_xlabel("lead time [days]")
    axes[0, 0].legend(fontsize=9, loc="best")
    fig.suptitle(f"Posterior conditioning — {tag}", y=0.995, fontsize=13)
    fig.tight_layout()
    out = PLOTS / f"metrics_compare_{tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Metrics plot → %s", out)


def metrics_table():
    rows = []
    tags = sorted({d.name.split("_", 1)[1] for d in RESULTS.glob("*_*")
                   if (d / "scores" / "metrics_summary.csv").exists()})
    # Drop transient/dev tags (smoke, chunktest, ...) from the published table.
    tags = [t for t in tags if t not in ("smoke", "chunktest")]
    for tag in tags:
        for m in METHODS_ORDER:
            f = RESULTS / f"{m}_{tag}" / "scores" / "metrics_summary.csv"
            if not f.exists():
                continue
            s = pd.read_csv(f, index_col=0).iloc[:, 0]
            rows.append({
                "tag": tag,
                "method": m,
                "rmse_mean": float(s.get("rmse_mean", float("nan"))),
                "crps": float(s.get("crps", float("nan"))),
                "spread": float(s.get("spread", float("nan"))),
                "spread_error_ratio": float(s.get("spread_error_ratio", float("nan"))),
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(PLOTS / "metrics_table.csv", index=False)
        logger.info("Metrics table:\n%s", df.to_string(index=False))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None,
                   help="Restrict to a single tag (e.g. 1month). Default: all available.")
    p.add_argument("--skip-animations", action="store_true")
    p.add_argument("--max-frames", type=int, default=180)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    available_tags = sorted({d.name.split("_", 1)[1]
                             for d in RESULTS.glob("*_*")
                             if (d / "preds_trajectory.zarr").exists()})
    tags = [args.tag] if args.tag else available_tags
    logger.info("Tags: %s", tags)

    for tag in tags:
        # Method comparison panels at +6h, +7d, +30d (if in range)
        # 1month has 120 leads (0..119), full has 1460 leads (0..1459)
        leads_compare = [1, 28, 119] if tag == "1month" else [1, 28, 119]
        comparison_at_leads(tag, leads=leads_compare, n_inits_show=5)
        # Per-method trajectory-evolution panels
        for m in METHODS_ORDER:
            trajectory_evolution_panel(m, tag,
                                       leads=[1, 28, 60, 119] if tag == "1month"
                                       else [1, 28, 365, 1459],
                                       n_samples_show=5)
        # Metrics plot per tag
        metrics_plot(tag)

        if args.skip_animations:
            continue
        for m in METHODS_ORDER:
            # ensemble-mean animation
            animate_method(m, tag, max_frames=args.max_frames, fps=args.fps)
            # individual sample animation
            animate_method(m, tag, individual_sample=0, individual_init=0,
                           max_frames=args.max_frames, fps=args.fps)

    metrics_table()


if __name__ == "__main__":
    main()
