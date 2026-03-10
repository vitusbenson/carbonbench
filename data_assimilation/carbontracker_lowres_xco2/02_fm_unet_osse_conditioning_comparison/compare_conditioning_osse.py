"""
OSSE conditioning comparison for flow matching conditional generation (L1 / XCO2).

Uses CarbonTracker L1 data as BOTH training distribution AND pseudo-observations,
providing ground truth for quantitative evaluation of conditioning methods.

For L1, the model predicts total column XCO2 directly (single level), so no
averaging kernel is needed. All masking is "simple" (direct replacement).
Column and 3D masking are equivalent for a single level.

Usage:
    CUDA_VISIBLE_DEVICES=7 python compare_conditioning_osse.py --mask_pattern random
    CUDA_VISIBLE_DEVICES=7 python compare_conditioning_osse.py --mask_pattern satellite
    CUDA_VISIBLE_DEVICES=7 python compare_conditioning_osse.py --experiments correction velocity_proj guidance_10
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

from neural_transport.datasets.grids import (
    LATLON_PROTOTYPE_COORDS,
    VERTICAL_LAYERS_PROTOTYPE_COORDS,
)
from neural_transport.datasets.vars import *  # noqa: F403
from neural_transport.inference.generative import iterative_generate
from neural_transport.training.train import NeuralTransport, load_dataset

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)

# ── paths ──────────────────────────────────────────────────────────────────
DEFAULT_TRAINING_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/tmp_vitus/Carbontracker"
DEFAULT_FORECAST_DATA_ROOT = DEFAULT_TRAINING_DATA_ROOT + "/test"

RUN_DIR = Path(__file__).resolve().parent
CKPT_DIR = RUN_DIR.parent / "01_fm_unet_oco2_l1" / "singlestep" / "checkpoints"
PRED_ROOT = RUN_DIR / "singlestep" / "predictions"
ANALYSIS_ROOT = RUN_DIR / "singlestep" / "analysis"

# ── model / data config (matches exp 01 L1) ───────────────────────────────
TARGET_VARS = ["co2massmix"]
grid = "latlon5.625"
vertical_levels = "l1"
freq = "6h"

nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[vertical_levels]["level"])  # 1
lat = LATLON_PROTOTYPE_COORDS[grid]["lat"]
lon = LATLON_PROTOTYPE_COORDS[grid]["lon"]

cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(len(lon), axis=1).reshape(-1, 1)
cos_lat = cos_lat / np.mean(cos_lat)

METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in ["co2massmix"]}

regulargrid_kwargs = dict(
    input_vars=TARGET_VARS,
    target_vars=TARGET_VARS,
    nlat=len(lat),
    nlon=len(lon),
    predict_delta=False,
    add_surfflux=False,
    dt=60 * 60 * 6,
    massfixer="",
    targshift=True,
)

wrapper_kwargs = dict(
    **regulargrid_kwargs,
    model_kwargs=dict(
        submodel="unet",
        model_kwargs=dict(
            **regulargrid_kwargs,
            model_kwargs=dict(
                in_chans=nlev + 1,  # 1 + 1 for flow_time
                out_chans=nlev,     # 1
                embed_dim=128,
                act="leakyrelu",
                norm="batch",
                enc_filters=[[7], [3, 3], [3, 3], [3, 3]],
                dec_filters=[[3, 3], [3, 3], [3, 3], [3, 3]],
                in_interpolation="bilinear",
                out_interpolation="nearest-exact",
                out_clip=None,
            ),
        ),
        generating=True,
        return_intermediates=True,
        method="midpoint",
        nlev=nlev,
        step_size=None,
    ),
)

lit_module_kwargs = dict(
    model="flowmatching",
    model_kwargs=wrapper_kwargs,
    loss="flowmatching_mse",
    loss_kwargs=dict(),
    metrics=[
        dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS))
        for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]
    ],
    no_grad_step_shedule=None,
    lr=1e-3,
    weight_decay=0.1,
    lr_shedule_kwargs=dict(
        warmup_steps=1000, halfcosine_steps=99000, min_lr=3e-7, max_lr=1.0
    ),
    val_dataloader_names=["singlestep"],
    plot_kwargs=dict(
        variables=["co2molemix"],
        layer_idxs=[0],
        n_samples=2,
        dataset="carbontracker",
        grid=grid,
        vertical_levels=vertical_levels,
        max_workers=32,
    ),
)

# CarbonTracker L1 data for BOTH training dataset and generation/masking
# No forcing_vars needed for L1 (no pressure levels, no AK)
data_kwargs = dict(
    data_path=DEFAULT_TRAINING_DATA_ROOT,
    dataset="carbontracker",
    grid=grid,
    vertical_levels=vertical_levels,
    freq=freq,
    n_timesteps=1,
    batch_size_train=64,
    batch_size_pred=32,
    num_workers=32,
    val_rollout_n_timesteps=None,
    target_vars=["co2massmix"],
    forcing_vars=[],
    compute=False,
)

# Same data for generation (OSSE: CarbonTracker is both source and target)
generate_data_kwargs = data_kwargs.copy()

# ── experiments to compare ────────────────────────────────────────────────
# For L1, column and 3D masking are equivalent (single level),
# so we only use "test" (direct field masking) with "simple" method.
EXPERIMENTS = {
    "unconditional": dict(
        _masking=False,
    ),
    "correction": dict(
        conditioning_mode="correction",
        masking_time=None,
    ),
    "correction_late": dict(
        conditioning_mode="correction",
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    "velocity_proj": dict(
        conditioning_mode="velocity_projection",
        masking_time=None,
    ),
    "velocity_proj_late": dict(
        conditioning_mode="velocity_projection",
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    "guidance_1": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_time=None,
    ),
    "guidance_10": dict(
        conditioning_mode="guidance",
        guidance_scale=10.0,
        masking_time=None,
    ),
    "repaint": dict(
        conditioning_mode="repaint",
        masking_time=None,
    ),
    "repaint_late": dict(
        conditioning_mode="repaint",
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
}


def build_generate_kwargs(exp_overrides, mask_pattern="random", mask_source="test"):
    """Build full generate_kwargs for an experiment."""
    masking = exp_overrides.pop("_masking", True)

    base = dict(
        n_samples=10,
        refine_start=0.9,
        steps=11,
        avg_over_levels=False,
        condition_one_timestep=True,
        data_path_generate=DEFAULT_TRAINING_DATA_ROOT + "/train",
        generate_data_kwargs=generate_data_kwargs,
        masking=masking,
        mask_source=mask_source,
        mask_pattern=mask_pattern,
        masking_time=None,
        t_threshold=0.9,
        masking_method="simple",
        analyze_masking=False,
        obs_fraction=0.3,
        noise_pattern=None,
        analyze_noise=False,
    )
    base.update(exp_overrides)
    return base


def load_model_and_data(device):
    """Load exp 01 L1 checkpoint and CarbonTracker L1 dataset."""
    bestckptpath = sorted(
        [p for p in CKPT_DIR.glob("*.ckpt") if "LossVal" in p.name],
        key=lambda p: float(p.name.split("=")[-1].split(".c")[0]),
    )[0]
    print(f"Using checkpoint: {bestckptpath.name}")

    model = NeuralTransport.load_from_checkpoint(bestckptpath, **lit_module_kwargs)

    data_path_forecast = Path(DEFAULT_FORECAST_DATA_ROOT)
    dataset = load_dataset(data_path_forecast, data_kwargs, load_obspack=False)

    return model, dataset


def run_experiment(name, model, dataset, generate_kwargs, device,
                   mask_source="test", mask_pattern="random"):
    """Run a single conditioning experiment."""
    outpath = PRED_ROOT / f"{mask_source}_{mask_pattern}" / name
    outpath.mkdir(parents=True, exist_ok=True)

    model_copy = copy.deepcopy(model)

    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    config_display = {k: v for k, v in generate_kwargs.items()
                      if k not in ('generate_data_kwargs', 'data_path_generate')}
    print(f"  Config: {json.dumps(config_display, default=str, indent=4)}")
    print(f"{'='*60}")

    ds_all = iterative_generate(
        model_copy,
        dataset,
        outpath,
        rollout=False,
        device=device,
        verbose=True,
        freq="QS",
        zero_surfflux=False,
        remap=False,
        target_vars_3d=data_kwargs["target_vars"],
        target_vars_2d=[],
        save_obs=False,
        **generate_kwargs,
    )

    return ds_all


def get_ground_truth(dataset, nlat, nlon):
    """Extract ground truth field from the first CarbonTracker timestep."""
    batch = {k: v.unsqueeze(0) for k, v in dataset[0].items()}
    gt = batch["co2massmix"]  # [1, 1, N, C]
    B, T, N, C = gt.shape
    gt_grid = gt.reshape(B, T, nlat, nlon, C)
    return gt_grid[0, 0].numpy()  # [nlat, nlon, C]


def _extract_mask_2d(ds_pred, nlat, nlon):
    """Extract 2D spatial mask from dataset, handling both column and 3D masks."""
    if "obs_mask" not in ds_pred:
        return None
    mask = ds_pred["obs_mask"]
    if "time" in mask.dims:
        mask = mask.isel(time=0)
    # Use isel(level=0) if level dimension exists, otherwise squeeze
    if "level" in mask.dims:
        mask_np = mask.isel(level=0).values.astype(bool)
    else:
        mask_np = mask.values.astype(bool).squeeze()
    # mask_np should be [N] or [nlat*nlon] at this point
    if mask_np.ndim > 1:
        mask_np = mask_np.flatten()
    n_cells = nlat * nlon
    if mask_np.size > n_cells:
        mask_np = mask_np[:n_cells]
    mask_2d = mask_np.reshape(nlat, nlon)
    return mask_2d


def compute_gt_metrics(ds_pred, gt_field, nlat, nlon):
    """Compute ground truth metrics for OSSE evaluation (single level).

    For L1, rmse_full == rmse_col since there is only one level.

    Args:
        ds_pred: xarray Dataset with predictions (has 'co2massmix' with sample dim)
        gt_field: numpy array [nlat, nlon, 1] ground truth field
        nlat, nlon: grid dimensions
    """
    pred = ds_pred["co2massmix"]
    if "trajectory_steps" in pred.dims:
        pred = pred.isel(trajectory_steps=-1)
    if "time" in pred.dims:
        pred = pred.isel(time=0)

    # pred shape: [sample, cell, level] -> reshape to [sample, nlat, nlon, level]
    pred_np = pred.values
    n_samples = pred_np.shape[0]
    if pred_np.ndim == 3:
        pred_np = pred_np.reshape(n_samples, nlat, nlon, -1)

    gt = gt_field  # [nlat, nlon, 1]

    # Ensemble mean
    ens_mean = pred_np.mean(axis=0)  # [nlat, nlon, 1]

    # Cos-lat weights [nlat, 1, 1]
    cos_w = np.cos(np.radians(lat))[:, None, None]
    cos_w = cos_w / cos_w.mean()

    # 3D field metrics (== column metrics for single level)
    diff = ens_mean - gt
    rmse_3d_full = float(np.sqrt(np.mean(cos_w * diff ** 2)))

    # R2
    ss_res = np.sum(cos_w * diff ** 2)
    gt_mean = np.mean(cos_w * gt) / np.mean(cos_w)
    ss_tot = np.sum(cos_w * (gt - gt_mean) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-12))

    # Obs-location RMSE
    rmse_3d_obs = np.nan
    rmse_3d_away = np.nan

    mask_2d = _extract_mask_2d(ds_pred, nlat, nlon)
    if mask_2d is not None:
        mask_3d = mask_2d[:, :, None].repeat(gt.shape[-1], axis=-1)

        if mask_3d.any():
            rmse_3d_obs = float(np.sqrt(np.mean(diff[mask_3d] ** 2)))
        if (~mask_3d).any():
            rmse_3d_away = float(np.sqrt(np.mean(diff[~mask_3d] ** 2)))

    # Spread-skill ratio
    ens_spread = pred_np.std(axis=0)
    ens_error = np.abs(diff)
    spread_skill = float(np.mean(ens_spread) / max(np.mean(ens_error), 1e-12))

    # Spatial roughness (striping proxy)
    pred_col = ens_mean.mean(axis=-1)  # [nlat, nlon]
    laplacian_lat = np.diff(pred_col, n=2, axis=0)
    roughness_lat = float(np.std(laplacian_lat))
    laplacian_lon = np.diff(pred_col, n=2, axis=1)
    roughness_lon = float(np.std(laplacian_lon))

    # Sample spread
    sample_spread = float(pred.std(dim="sample").mean().values)

    return {
        "rmse_3d_full": rmse_3d_full,
        "rmse_3d_obs": rmse_3d_obs,
        "rmse_3d_away": rmse_3d_away,
        "r2": r2,
        "spread_skill": spread_skill,
        "roughness_lat": roughness_lat,
        "roughness_lon": roughness_lon,
        "sample_spread": sample_spread,
        # Legacy aliases (compatible with exp 08 metric keys)
        "rmse_full": rmse_3d_full,
        "rmse_obs": rmse_3d_obs,
        "rmse_away": rmse_3d_away,
    }


def plot_comparison(results, gt_field, out_dir, nlat, nlon):
    """Create a comparison figure: rows=experiments, cols=[GT, Observed, Ens Mean, |Diff|, Sample0..2]."""
    n_exp = len(results)
    fig, axes = plt.subplots(
        n_exp, 7, figsize=(28, 3.5 * n_exp),
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    if n_exp == 1:
        axes = axes[np.newaxis, :]

    # Single level -> squeeze to 2D
    gt_col = gt_field.mean(axis=-1)  # [nlat, nlon]

    for row, (name, res) in enumerate(results.items()):
        ds = res["ds"]
        metrics = res["metrics"]
        pred = ds["co2massmix"]
        if "trajectory_steps" in pred.dims:
            pred = pred.isel(trajectory_steps=-1)
        if "time" in pred.dims:
            pred = pred.isel(time=0)
        if "level" in pred.dims:
            pred = pred.mean(dim="level")

        samples = pred.values  # [sample, ...]
        if samples.ndim == 2:
            samples = samples.reshape(-1, nlat, nlon)
        elif samples.ndim == 3 and samples.shape[-1] != nlon:
            samples = samples.reshape(-1, nlat, nlon)

        ens_mean = np.nanmean(samples, axis=0)
        diff = np.abs(ens_mean - gt_col)

        vmin = np.nanpercentile(gt_col, 2)
        vmax = np.nanpercentile(gt_col, 98)

        # Col 0: Ground truth
        axes[row, 0].imshow(gt_col, origin="lower", cmap="cividis",
                            vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 0].set_title("Ground Truth" if row == 0 else "", fontsize=10)

        # Col 1: Conditioning info (GT at observed locations, unobserved white)
        mask_2d = _extract_mask_2d(ds, nlat, nlon)
        if mask_2d is not None:
            obs_display = np.where(mask_2d, gt_col, np.nan)
            axes[row, 1].imshow(obs_display, origin="lower", cmap="cividis",
                                vmin=vmin, vmax=vmax, aspect="auto")
        else:
            axes[row, 1].text(0.5, 0.5, "No obs", transform=axes[row, 1].transAxes,
                              ha="center", va="center", fontsize=10, color="gray")
        axes[row, 1].set_title("Observed" if row == 0 else "", fontsize=10)

        # Col 2: Ensemble mean
        axes[row, 2].imshow(ens_mean, origin="lower", cmap="cividis",
                            vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 2].set_title("Ens. Mean" if row == 0 else "", fontsize=10)

        # Col 3: Difference
        dmax = np.nanpercentile(diff, 98)
        axes[row, 3].imshow(diff, origin="lower", cmap="Reds",
                            vmin=0, vmax=dmax, aspect="auto")
        axes[row, 3].set_title("|Difference|" if row == 0 else "", fontsize=10)

        # Cols 4-6: Samples
        n_show = min(3, samples.shape[0])
        for i in range(n_show):
            ax = axes[row, 4 + i]
            ax.imshow(samples[i], origin="lower", cmap="cividis",
                      vmin=vmin, vmax=vmax, aspect="auto")
            if row == 0:
                ax.set_title(f"Sample {i}", fontsize=10)

        # Row label
        label = name.replace("_", "\n")
        metrics_str = (f"RMSE={metrics['rmse_3d_full']:.3f}\n"
                       f"R2={metrics['r2']:.3f}\n"
                       f"spread={metrics['sample_spread']:.2f}")
        axes[row, 0].set_ylabel(f"{label}\n\n{metrics_str}", fontsize=8,
                                 rotation=0, labelpad=100, va="center")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(n_show + 4, 7):
            axes[row, i].axis("off")

    fig.suptitle("OSSE Conditioning Comparison - L1 XCO2 (CarbonTracker GT)", fontsize=14, y=1.01)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(out_dir / f"conditioning_comparison.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison figure saved to {out_dir}/conditioning_comparison.png")


def plot_metrics_bar(results, out_dir):
    """Bar chart comparing GT metrics across experiments."""
    names = list(results.keys())
    metrics_keys = ["rmse_3d_full", "rmse_3d_obs", "rmse_3d_away",
                    "r2", "spread_skill", "roughness_lat", "roughness_lon",
                    "sample_spread"]
    labels = ["RMSE\n(full)", "RMSE\n(obs)", "RMSE\n(away)",
              "R\u00b2", "Spread/\nSkill", "Roughness\n(lat)", "Roughness\n(lon)",
              "Sample\nSpread"]

    n_metrics = len(metrics_keys)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    x = np.arange(len(names))

    for i, (key, label) in enumerate(zip(metrics_keys, labels)):
        values = []
        for n in names:
            v = results[n]["metrics"].get(key, np.nan)
            values.append(v if not (isinstance(v, float) and np.isnan(v)) else 0)
        bars = axes[i].bar(x, values, color=plt.cm.tab10(x / max(len(names), 1)))
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([n.replace("_", "\n") for n in names],
                                 fontsize=6, rotation=45, ha="right")
        axes[i].set_title(label, fontsize=11)
        axes[i].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    for i in range(n_metrics, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(out_dir / f"conditioning_metrics.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics bar chart saved to {out_dir}/conditioning_metrics.png")


def plot_zonal_mean(results, gt_field, out_dir, nlat, nlon):
    """Zonal mean line plot: latitude vs XCO2 for GT and each experiment.

    For L1 (single level), this is a simple 1D line plot instead of the
    lat-vs-level heatmap used in l10.
    """
    n_exp = len(results)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # GT zonal mean (single level)
    gt_zonal = gt_field[:, :, 0].mean(axis=1)  # [nlat]
    ax.plot(lat, gt_zonal, "k-", linewidth=2, label="Ground Truth")

    colors = plt.cm.tab10(np.linspace(0, 1, n_exp))
    for i, (name, res) in enumerate(results.items()):
        ds = res["ds"]
        pred = ds["co2massmix"]
        if "trajectory_steps" in pred.dims:
            pred = pred.isel(trajectory_steps=-1)
        if "time" in pred.dims:
            pred = pred.isel(time=0)

        pred_mean = pred.mean(dim="sample").values
        if pred_mean.ndim == 2:
            # [cell, level] -> [nlat, nlon]
            pred_mean = pred_mean[:, 0].reshape(nlat, nlon)
        elif pred_mean.ndim == 1:
            pred_mean = pred_mean.reshape(nlat, nlon)
        pred_zonal = pred_mean.mean(axis=1)  # [nlat]

        ax.plot(lat, pred_zonal, color=colors[i], linewidth=1.2, label=name)

    ax.set_xlabel("Latitude")
    ax.set_ylabel("XCO2 (co2massmix)")
    ax.set_title("Zonal Mean XCO2 - OSSE Comparison")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(out_dir / f"zonal_mean_comparison.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Zonal mean plot saved to {out_dir}/zonal_mean_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_pattern", type=str, default="random",
                        choices=["random", "satellite", "checkerboard", "vertical", "horizontal"])
    parser.add_argument("--mask_source", type=str, default="test",
                        choices=["test"],
                        help="For L1, only 'test' (direct field masking) is supported")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Subset of experiments to run (default: all)")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--obs_fraction", type=float, default=0.3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    analysis_dir = ANALYSIS_ROOT / f"{args.mask_source}_{args.mask_pattern}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load model + dataset once
    model, dataset = load_model_and_data(device)

    # Ground truth
    nlat_val, nlon_val = len(lat), len(lon)
    gt_field = get_ground_truth(dataset, nlat_val, nlon_val)

    # Select experiments
    all_experiments = EXPERIMENTS
    if args.experiments:
        experiments = {k: v for k, v in all_experiments.items() if k in args.experiments}
        if not experiments:
            print(f"No matching experiments. Available: {list(all_experiments.keys())}")
            sys.exit(1)
    else:
        experiments = all_experiments

    results = {}

    for name, overrides in experiments.items():
        overrides = copy.deepcopy(overrides)
        generate_kwargs = build_generate_kwargs(
            overrides, mask_pattern=args.mask_pattern,
            mask_source=args.mask_source)
        generate_kwargs["n_samples"] = args.n_samples
        generate_kwargs["obs_fraction"] = args.obs_fraction

        pl.seed_everything(42)

        try:
            ds_all = run_experiment(
                name, model, dataset, generate_kwargs, device,
                mask_source=args.mask_source, mask_pattern=args.mask_pattern)
            metrics = compute_gt_metrics(ds_all, gt_field, nlat_val, nlon_val)
            results[name] = {"ds": ds_all, "metrics": metrics}
            print(f"  -> Metrics: {metrics}")
        except Exception as e:
            import traceback
            print(f"  !! FAILED: {e}")
            traceback.print_exc()
            continue

    if not results:
        print("No experiments succeeded!")
        sys.exit(1)

    # Plots
    plot_comparison(results, gt_field, analysis_dir, nlat_val, nlon_val)
    plot_metrics_bar(results, analysis_dir)
    plot_zonal_mean(results, gt_field, analysis_dir, nlat_val, nlon_val)

    # Save metrics
    summary = {name: res["metrics"] for name, res in results.items()}
    with open(analysis_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nMetrics summary saved to {analysis_dir / 'metrics_summary.json'}")

    # Print table
    print(f"\n{'='*120}")
    header = (f"{'Experiment':<20} {'RMSE_full':>10} {'RMSE_obs':>10} {'RMSE_away':>10} "
              f"{'R2':>8} {'Spr/Skl':>8} {'Rough_lat':>10} {'Rough_lon':>10} {'Spread':>10}")
    print(header)
    print(f"{'-'*120}")
    for name, res in results.items():
        m = res["metrics"]
        def _fmt(v):
            return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
        print(f"{name:<20} {_fmt(m['rmse_3d_full']):>10} {_fmt(m['rmse_3d_obs']):>10} {_fmt(m['rmse_3d_away']):>10} "
              f"{_fmt(m['r2']):>8} {m['spread_skill']:>8.3f} {m['roughness_lat']:>10.4f} "
              f"{m['roughness_lon']:>10.4f} {m['sample_spread']:>10.4f}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
