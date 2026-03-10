"""
OSSE conditioning comparison for flow matching conditional generation.

Uses CarbonTracker data as BOTH training distribution AND pseudo-observations,
providing ground truth for quantitative evaluation of conditioning methods.

Uses exp 07 checkpoint with CarbonTracker data (like exp 06) and a clean
comparison structure (like exp 07's compare_conditioning.py).

Usage:
    CUDA_VISIBLE_DEVICES=7 python compare_conditioning_osse.py --mask_source xco2 --mask_pattern random
    CUDA_VISIBLE_DEVICES=7 python compare_conditioning_osse.py --mask_source 3d --mask_pattern satellite
    CUDA_VISIBLE_DEVICES=7 python compare_conditioning_osse.py --experiments correction velocity_proj guidance_10
    CUDA_VISIBLE_DEVICES=7 python compare_conditioning_osse.py --migrate  # migrate old results to new folder structure
"""

import argparse
import copy
import json
import shutil
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
DEFAULT_TRAINING_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
DEFAULT_FORECAST_DATA_ROOT = DEFAULT_TRAINING_DATA_ROOT + "/test"

RUN_DIR = Path(__file__).resolve().parent
CKPT_DIR = RUN_DIR.parent / "07_fm_unet_oco2" / "singlestep" / "checkpoints"
PRED_ROOT = RUN_DIR / "singlestep" / "predictions"
ANALYSIS_ROOT = RUN_DIR / "singlestep" / "analysis"

# Legacy path (for migration)
OLD_OUT_ROOT = RUN_DIR / "singlestep" / "conditioning_comparison"

# ── model / data config (matches exp 07) ───────────────────────────────────
TARGET_VARS = ["co2massmix"]
grid = "latlon5.625"
vertical_levels = "l10"
freq = "6h"

nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[vertical_levels]["level"])
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
                in_chans=nlev + 1,
                out_chans=nlev,
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
        layer_idxs=[0, 3, 5, 8],
        n_samples=2,
        dataset="carbontracker",
        grid=grid,
        vertical_levels=vertical_levels,
        max_workers=32,
    ),
)

# CarbonTracker data for BOTH training dataset and generation/masking
# forcing_vars includes p_bottom/p_top for computing pressure weights in column OSSE
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
    forcing_vars=["p_bottom", "p_top"],
    compute=False,
)

# Same data for generation (OSSE: CarbonTracker is both source and target)
generate_data_kwargs = data_kwargs.copy()


def get_mean_ak_on_l10():
    """Compute mean OCO-2 averaging kernel aggregated to CarbonTracker l10 levels.

    Takes the 34-level mean AK from carbontracker.py and aggregates to 10 levels
    using the same level groupings as CARBONTRACKER_LEVEL_AGG['l10'].
    """
    from neural_transport.datasets.carbontracker import (
        get_mean_oco2_ak_on_ct_levels,
        CARBONTRACKER_LEVEL_AGG,
    )
    # AK on 34 CT levels
    data_root = Path(DEFAULT_TRAINING_DATA_ROOT).parent
    ak_34 = get_mean_oco2_ak_on_ct_levels(data_root)

    # Aggregate to l10 using simple mean within each group
    # (proper aggregation would use pressure-weighted mean, but since AK
    #  varies slowly and groups are small, simple mean is adequate)
    level_groups = CARBONTRACKER_LEVEL_AGG["l10"]
    ak_10 = np.array([ak_34[group].mean() for group in level_groups])
    return ak_10

# ── experiments to compare ────────────────────────────────────────────────
# 3D OSSE: all methods work (direct field masking, no spatial discontinuity issues)
EXPERIMENTS_3D = {
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

# Column OSSE: only late-masking and weak-guidance variants are stable.
# The correction mode and strong early conditioning create spatial discontinuities
# (uniform level shift at obs locations vs noise elsewhere) that the UNet can't handle.
EXPERIMENTS_COLUMN = {
    "unconditional": dict(
        _masking=False,
    ),
    "velocity_proj_late": dict(
        conditioning_mode="velocity_projection",
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    "guidance_0.5": dict(
        conditioning_mode="guidance",
        guidance_scale=0.5,
        masking_time=None,
    ),
    "guidance_1": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_time=None,
    ),
    "guidance_1_late": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    "guidance_3_late": dict(
        conditioning_mode="guidance",
        guidance_scale=3.0,
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    "repaint_late": dict(
        conditioning_mode="repaint",
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
}

# Legacy alias
EXPERIMENTS = EXPERIMENTS_3D


def build_generate_kwargs(exp_overrides, mask_pattern="random", mask_source="3d",
                          ak_10=None):
    """Build full generate_kwargs for an experiment."""
    masking = exp_overrides.pop("_masking", True)

    # Use appropriate masking method:
    # - xco2 OSSE: total_column_average_simple (column-level correction via AK)
    # - 3d OSSE: simple (direct replacement at observed grid cells)
    if mask_source == "xco2":
        masking_method = "total_column_average_simple"
    else:
        masking_method = "simple"

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
        masking_method=masking_method,
        analyze_masking=False,
        obs_fraction=0.3,
        noise_pattern=None,
        analyze_noise=False,
        ak_10=ak_10,
    )
    base.update(exp_overrides)
    return base


def load_model_and_data(device):
    """Load exp 07 checkpoint and CarbonTracker dataset."""
    bestckptpath = sorted(
        [p for p in CKPT_DIR.glob("*.ckpt") if "LossVal" in p.name],
        key=lambda p: float(p.name.split("=")[-1].split(".c")[0]),
    )[0]
    print(f"Using checkpoint: {bestckptpath.name}")

    model = NeuralTransport.load_from_checkpoint(bestckptpath, **lit_module_kwargs)

    data_path_forecast = Path(DEFAULT_FORECAST_DATA_ROOT)
    dataset = load_dataset(data_path_forecast, data_kwargs)

    return model, dataset


def run_experiment(name, model, dataset, generate_kwargs, device,
                   mask_source="3d", mask_pattern="random"):
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


def get_pressure_weights_and_ak(dataset, nlat, nlon, ak_10):
    """Get pressure weights and AK for XCO2 computation from ground truth data."""
    batch = {k: v.unsqueeze(0) for k, v in dataset[0].items()}
    p_bottom = batch["p_bottom"]  # [1, 1, N, C]
    p_top = batch["p_top"]        # [1, 1, N, C]
    dp = p_bottom - p_top
    p_surface = p_bottom[:, :, :, 0:1]
    h_k = (dp / p_surface.clamp(min=1e-6))[0, 0].numpy()  # [N, C]
    h_k = h_k.reshape(nlat, nlon, -1)  # [nlat, nlon, C]

    C = h_k.shape[-1]
    if ak_10 is not None:
        ak = np.broadcast_to(ak_10.reshape(1, 1, C), (nlat, nlon, C)).copy()
    else:
        ak = np.ones((nlat, nlon, C))

    return h_k, ak


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


def compute_gt_metrics(ds_pred, gt_field, nlat, nlon,
                       pressure_weights=None, ak_weights=None):
    """Compute ground truth metrics for OSSE evaluation.

    Args:
        ds_pred: xarray Dataset with predictions (has 'co2massmix' with sample dim)
        gt_field: numpy array [nlat, nlon, C] ground truth field
        nlat, nlon: grid dimensions
        pressure_weights: numpy array [nlat, nlon, C] h_k weights (optional, for XCO2 metrics)
        ak_weights: numpy array [nlat, nlon, C] averaging kernel (optional, for XCO2 metrics)
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

    gt = gt_field  # [nlat, nlon, C]

    # Ensemble mean
    ens_mean = pred_np.mean(axis=0)  # [nlat, nlon, C]

    # Cos-lat weights [nlat, 1, 1]
    cos_w = np.cos(np.radians(lat))[:, None, None]
    cos_w = cos_w / cos_w.mean()

    # 3D field metrics
    diff = ens_mean - gt
    rmse_3d_full = float(np.sqrt(np.mean(cos_w * diff ** 2)))

    # Column-mean RMSE (average over levels first)
    diff_col = diff.mean(axis=-1)  # [nlat, nlon]
    cos_w_2d = cos_w[:, :, 0]
    rmse_col = float(np.sqrt(np.mean(cos_w_2d * diff_col ** 2)))

    # R2
    ss_res = np.sum(cos_w * diff ** 2)
    gt_mean = np.mean(cos_w * gt) / np.mean(cos_w)
    ss_tot = np.sum(cos_w * (gt - gt_mean) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-12))

    # Obs-location RMSE (3D field, if mask available)
    rmse_3d_obs = np.nan
    rmse_3d_away = np.nan
    rmse_xco2_obs = np.nan
    rmse_xco2_away = np.nan
    rmse_xco2_full = np.nan

    mask_2d = _extract_mask_2d(ds_pred, nlat, nlon)
    if mask_2d is not None:
        # Expand to 3D for indexing diff [nlat, nlon, C]
        mask_3d = mask_2d[:, :, None].repeat(gt.shape[-1], axis=-1)

        if mask_3d.any():
            rmse_3d_obs = float(np.sqrt(np.mean(diff[mask_3d] ** 2)))
        if (~mask_3d).any():
            rmse_3d_away = float(np.sqrt(np.mean(diff[~mask_3d] ** 2)))

    # XCO2 metrics (total column)
    if pressure_weights is not None and ak_weights is not None:
        h_ak = pressure_weights * ak_weights  # [nlat, nlon, C]
        xco2_pred = (h_ak * ens_mean).sum(axis=-1)  # [nlat, nlon]
        xco2_gt = (h_ak * gt).sum(axis=-1)  # [nlat, nlon]
        xco2_diff = xco2_pred - xco2_gt

        rmse_xco2_full = float(np.sqrt(np.mean(cos_w_2d * xco2_diff ** 2)))

        if mask_2d is not None:
            if mask_2d.any():
                rmse_xco2_obs = float(np.sqrt(np.mean(xco2_diff[mask_2d] ** 2)))
            if (~mask_2d).any():
                rmse_xco2_away = float(np.sqrt(np.mean(xco2_diff[~mask_2d] ** 2)))

    # Spread-skill ratio
    ens_spread = pred_np.std(axis=0)  # [nlat, nlon, C]
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
        "rmse_xco2_full": rmse_xco2_full,
        "rmse_xco2_obs": rmse_xco2_obs,
        "rmse_xco2_away": rmse_xco2_away,
        "rmse_col": rmse_col,
        "r2": r2,
        "spread_skill": spread_skill,
        "roughness_lat": roughness_lat,
        "roughness_lon": roughness_lon,
        "sample_spread": sample_spread,
        # Legacy aliases
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

    fig.suptitle("OSSE Conditioning Comparison (CarbonTracker GT)", fontsize=14, y=1.01)
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
                    "rmse_xco2_full", "rmse_xco2_obs", "rmse_xco2_away",
                    "r2", "spread_skill", "roughness_lat", "roughness_lon",
                    "sample_spread"]
    labels = ["RMSE 3D\n(full)", "RMSE 3D\n(obs)", "RMSE 3D\n(away)",
              "RMSE XCO2\n(full)", "RMSE XCO2\n(obs)", "RMSE XCO2\n(away)",
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
    """Zonal mean profiles: lat vs level for GT and each experiment."""
    n_exp = len(results)
    fig, axes = plt.subplots(1, n_exp + 1, figsize=(4 * (n_exp + 1), 5),
                              sharey=True)

    gt_zonal = gt_field.mean(axis=1)  # [nlat, C]
    vmin = np.nanpercentile(gt_zonal, 2)
    vmax = np.nanpercentile(gt_zonal, 98)

    axes[0].imshow(gt_zonal, origin="lower", cmap="cividis",
                   vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("Ground Truth", fontsize=10)
    axes[0].set_ylabel("Latitude index")
    axes[0].set_xlabel("Level")

    for i, (name, res) in enumerate(results.items()):
        ds = res["ds"]
        pred = ds["co2massmix"]
        if "trajectory_steps" in pred.dims:
            pred = pred.isel(trajectory_steps=-1)
        if "time" in pred.dims:
            pred = pred.isel(time=0)

        pred_mean = pred.mean(dim="sample").values
        if pred_mean.ndim == 2:
            pred_mean = pred_mean.reshape(nlat, nlon, -1)
        pred_zonal = pred_mean.mean(axis=1)  # [nlat, C]

        axes[i + 1].imshow(pred_zonal, origin="lower", cmap="cividis",
                           vmin=vmin, vmax=vmax, aspect="auto")
        axes[i + 1].set_title(name, fontsize=9)
        axes[i + 1].set_xlabel("Level")

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(out_dir / f"zonal_mean_comparison.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Zonal mean plot saved to {out_dir}/zonal_mean_comparison.png")


def migrate_old_results():
    """Migrate old flat directory structure to new predictions/analysis layout.

    Old layout (under singlestep/conditioning_comparison/):
        test_random/, test_satellite/, test_checkerboard/ -> 3d_{pattern}/
        column_random/, column_satellite/, column_checkerboard/ -> xco2_{pattern}/
        correction/, guidance_1/, etc. (flat experiment dirs) -> 3d_random/{name}/
        col_* dirs -> xco2_random/{name_without_col_prefix}/
        3d_* dirs -> 3d_random/{name}/
    """
    if not OLD_OUT_ROOT.exists():
        print(f"No old results found at {OLD_OUT_ROOT}")
        return

    moved = 0

    for item in sorted(OLD_OUT_ROOT.iterdir()):
        if not item.is_dir():
            continue
        name = item.name

        # Pattern: {mask_source}_{mask_pattern} directories containing experiment subdirs
        # e.g. test_random/, column_satellite/
        if name.startswith("test_") or name.startswith("column_"):
            parts = name.split("_", 1)
            old_source = parts[0]
            pattern = parts[1]
            new_source = "3d" if old_source == "test" else "xco2"

            # Check if this is a compound dir (contains experiment subdirs or analysis files)
            has_zarr = any(item.glob("*.zarr"))
            has_json = any(item.glob("*.json"))
            has_png = any(item.glob("*.png"))

            if has_json or has_png:
                # This is an analysis output directory - move analysis files
                dest_analysis = ANALYSIS_ROOT / f"{new_source}_{pattern}"
                dest_analysis.mkdir(parents=True, exist_ok=True)
                for f in item.iterdir():
                    if f.is_file() and f.suffix in (".json", ".png", ".pdf"):
                        shutil.copy2(f, dest_analysis / f.name)
                        moved += 1

            # Check for experiment subdirs
            for subdir in item.iterdir():
                if subdir.is_dir():
                    dest = PRED_ROOT / f"{new_source}_{pattern}" / subdir.name
                    if not dest.exists():
                        dest.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(subdir, dest, dirs_exist_ok=True)
                        moved += 1
                        print(f"  {name}/{subdir.name} -> predictions/{new_source}_{pattern}/{subdir.name}")

        # Flat experiment dirs (from default 3d_random runs)
        elif name.startswith("col_"):
            # Column experiment: col_guidance_1 -> xco2_random/guidance_1
            exp_name = name[4:]  # strip "col_" prefix
            dest = PRED_ROOT / "xco2_random" / exp_name
            if not dest.exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(item, dest, dirs_exist_ok=True)
                moved += 1
                print(f"  {name} -> predictions/xco2_random/{exp_name}")

        elif name.startswith("3d_"):
            # Explicit 3D experiment: 3d_correction -> 3d_random/correction
            exp_name = name[3:]  # strip "3d_" prefix
            dest = PRED_ROOT / "3d_random" / exp_name
            if not dest.exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(item, dest, dirs_exist_ok=True)
                moved += 1
                print(f"  {name} -> predictions/3d_random/{exp_name}")

        else:
            # Bare experiment name (correction, guidance_1, etc.) -> 3d_random/{name}
            # Only if it contains zarr files (i.e., is actually a prediction dir)
            if any(item.glob("*.zarr")) or any(item.glob("co2_pred_*.zarr")):
                dest = PRED_ROOT / "3d_random" / name
                if not dest.exists():
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                    moved += 1
                    print(f"  {name} -> predictions/3d_random/{name}")

    print(f"\nMigration complete: {moved} items moved")
    print(f"  Predictions: {PRED_ROOT}")
    print(f"  Analysis:    {ANALYSIS_ROOT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_pattern", type=str, default="random",
                        choices=["random", "satellite", "checkerboard", "vertical", "horizontal"])
    parser.add_argument("--mask_source", type=str, default="xco2",
                        choices=["3d", "xco2"],
                        help="'3d': 3D field masking, 'xco2': synthetic XCO2 column obs")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Subset of experiments to run (default: all)")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--obs_fraction", type=float, default=0.3)
    parser.add_argument("--migrate", action="store_true",
                        help="Migrate old results to new folder structure and exit")
    args = parser.parse_args()

    if args.migrate:
        migrate_old_results()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    analysis_dir = ANALYSIS_ROOT / f"{args.mask_source}_{args.mask_pattern}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Compute mean AK on l10 levels for column OSSE
    ak_10 = None
    if args.mask_source == "xco2":
        ak_10 = get_mean_ak_on_l10()
        print(f"Mean AK on l10 levels: {ak_10}")

    # Load model + dataset once
    model, dataset = load_model_and_data(device)

    # Ground truth
    nlat_val, nlon_val = len(lat), len(lon)
    gt_field = get_ground_truth(dataset, nlat_val, nlon_val)

    # Pressure weights and AK for XCO2 metrics
    pressure_weights, ak_weights = get_pressure_weights_and_ak(
        dataset, nlat_val, nlon_val, ak_10)

    # Select experiments based on mask_source
    all_experiments = EXPERIMENTS_COLUMN if args.mask_source == "xco2" else EXPERIMENTS_3D
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
            mask_source=args.mask_source, ak_10=ak_10)
        generate_kwargs["n_samples"] = args.n_samples
        generate_kwargs["obs_fraction"] = args.obs_fraction

        pl.seed_everything(42)

        try:
            ds_all = run_experiment(
                name, model, dataset, generate_kwargs, device,
                mask_source=args.mask_source, mask_pattern=args.mask_pattern)
            metrics = compute_gt_metrics(
                ds_all, gt_field, nlat_val, nlon_val,
                pressure_weights=pressure_weights, ak_weights=ak_weights)
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
    print(f"\n{'='*160}")
    header = (f"{'Experiment':<20} {'RMSE_3d':>10} {'RMSE_3d_o':>10} {'RMSE_3d_a':>10} "
              f"{'RMSE_xco2':>10} {'RMSE_xo':>10} {'RMSE_xa':>10} "
              f"{'R2':>8} {'Spr/Skl':>8} {'Rough_lat':>10} {'Rough_lon':>10} {'Spread':>10}")
    print(header)
    print(f"{'-'*160}")
    for name, res in results.items():
        m = res["metrics"]
        def _fmt(v):
            return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
        print(f"{name:<20} {_fmt(m['rmse_3d_full']):>10} {_fmt(m['rmse_3d_obs']):>10} {_fmt(m['rmse_3d_away']):>10} "
              f"{_fmt(m['rmse_xco2_full']):>10} {_fmt(m['rmse_xco2_obs']):>10} {_fmt(m['rmse_xco2_away']):>10} "
              f"{_fmt(m['r2']):>8} {m['spread_skill']:>8.3f} {m['roughness_lat']:>10.4f} "
              f"{m['roughness_lon']:>10.4f} {m['sample_spread']:>10.4f}")
    print(f"{'='*160}")


if __name__ == "__main__":
    main()
