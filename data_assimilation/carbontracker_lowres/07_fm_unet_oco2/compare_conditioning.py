"""
Compare different conditioning modes for flow matching conditional generation.

Runs the same OCO-2 masked generation with different conditioning strategies
and produces a side-by-side comparison figure + metrics table.

Usage:
    CUDA_VISIBLE_DEVICES=6 python compare_conditioning.py
"""

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
from neural_transport.inference.generative import iterative_generate_oco2
from neural_transport.training.train import NeuralTransport, load_dataset

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)

# ── paths ──────────────────────────────────────────────────────────────────
DEFAULT_TRAINING_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
DEFAULT_MASKING_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/OCO2MIP_OCO2"
DEFAULT_FORECAST_DATA_ROOT = DEFAULT_TRAINING_DATA_ROOT + "/test"

RUN_DIR = Path(__file__).resolve().parent
CKPT_DIR = RUN_DIR / "singlestep" / "checkpoints"
OUT_ROOT = RUN_DIR / "singlestep" / "conditioning_comparison"

# ── model / data config (same as train.py) ────────────────────────────────
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

generate_data_kwargs = dict(
    data_path=DEFAULT_MASKING_DATA_ROOT,
    dataset="mip_oco2",
    grid=grid,
    vertical_levels=vertical_levels,
    freq=freq,
    n_timesteps=1,
    batch_size_pred=32,
    num_workers=32,
    val_rollout_n_timesteps=None,
    target_vars=["xco2_2019_scale"],
    forcing_vars=[
        "xco2_averaging_kernel",
        "xco2_apriori",
        "co2_profile_apriori",
    ],
    compute=False,
)

# ── experiments to compare ────────────────────────────────────────────────
EXPERIMENTS = {
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
    "guidance_10": dict(
        conditioning_mode="guidance",
        guidance_scale=10.0,
        masking_time=None,
    ),
    "guidance_50": dict(
        conditioning_mode="guidance",
        guidance_scale=50.0,
        masking_time=None,
    ),
    "guidance_50_late": dict(
        conditioning_mode="guidance",
        guidance_scale=50.0,
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    "unconditional": dict(
        _masking=False,
    ),
}


def build_generate_kwargs(exp_overrides):
    """Build full generate_kwargs for an experiment."""
    masking = exp_overrides.pop("_masking", True)

    base = dict(
        n_samples=10,
        refine_start=0.9,
        steps=11,
        avg_over_levels=False,
        data_path_generate=DEFAULT_MASKING_DATA_ROOT + "/train",
        generate_data_kwargs=generate_data_kwargs,
        masking=masking,
        mask_source="oco2",
        mask_pattern=None,
        window_hours=24,
        masking_time=None,
        t_threshold=0.9,
        masking_method="total_column_average_simple_unitary",
        analyze_masking=False,
        obs_fraction=0.3,
        noise_pattern=None,
        analyze_noise=False,
    )
    base.update(exp_overrides)
    return base


def load_model_and_data(device):
    """Load checkpoint, forecast dataset, and OCO-2 dataset once."""
    bestckptpath = sorted(
        [p for p in CKPT_DIR.glob("*.ckpt") if "LossVal" in p.name],
        key=lambda p: float(p.name.split("=")[-1].split(".c")[0]),
    )[0]
    print(f"Using checkpoint: {bestckptpath.name}")

    model = NeuralTransport.load_from_checkpoint(bestckptpath, **lit_module_kwargs)

    data_path_forecast = Path(DEFAULT_FORECAST_DATA_ROOT)
    dataset = load_dataset(data_path_forecast, data_kwargs)
    dataset_gen = load_dataset(
        generate_data_kwargs["data_path"] + "/train",
        generate_data_kwargs,
    )

    return model, dataset, dataset_gen


def run_experiment(name, model, dataset, dataset_gen, generate_kwargs, device):
    """Run a single conditioning experiment and return the output zarr path."""
    outpath = OUT_ROOT / name
    outpath.mkdir(parents=True, exist_ok=True)

    # Clone model to avoid state leakage between experiments
    model_copy = copy.deepcopy(model)

    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  Config: {json.dumps({k: v for k, v in generate_kwargs.items() if k not in ('generate_data_kwargs', 'data_path_generate')}, default=str, indent=4)}")
    print(f"{'='*60}")

    masking = generate_kwargs.get("masking", True)

    if masking:
        ds_all = iterative_generate_oco2(
            model_copy,
            dataset,
            dataset_gen,
            outpath,
            rollout=False,
            device=device,
            verbose=True,
            freq="QS",
            zero_surfflux=False,
            remap=False,
            target_vars_3d=data_kwargs["target_vars"],
            target_vars_2d=generate_data_kwargs["target_vars"],
            save_obs=False,
            **generate_kwargs,
        )
    else:
        from neural_transport.inference.generative import iterative_generate
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


def compute_metrics(ds_all, varname="co2massmix"):
    """Compute self-contained metrics for a single experiment."""
    pred = ds_all[varname]
    if "trajectory_steps" in pred.dims:
        pred = pred.isel(trajectory_steps=-1)

    # Ensemble mean
    pred_mean = pred.mean(dim="sample")

    pred_np = pred_mean.values
    if pred_np.ndim == 4:  # [time, lat, lon, level]
        pred_np = pred_np.mean(axis=-1)

    # Spatial roughness: std of 2nd-order lat differences (striping proxy)
    laplacian_lat = np.diff(pred_np, n=2, axis=-2)
    spatial_roughness = float(np.std(laplacian_lat))

    # Spatial roughness along lon too
    laplacian_lon = np.diff(pred_np, n=2, axis=-1)
    spatial_roughness_lon = float(np.std(laplacian_lon))

    # Sample spread (ensemble diversity)
    sample_spread = float(pred.std(dim="sample").mean().values)

    # Global mean and std of ensemble mean
    global_mean = float(pred_np.mean())
    global_std = float(pred_np.std())

    return {
        "roughness_lat": spatial_roughness,
        "roughness_lon": spatial_roughness_lon,
        "sample_spread": sample_spread,
        "global_mean": global_mean,
        "global_std": global_std,
    }


def plot_comparison(results, out_dir):
    """Create a comparison figure across all experiments."""
    n_exp = len(results)
    fig, axes = plt.subplots(
        n_exp, 6, figsize=(24, 3.5 * n_exp),
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    if n_exp == 1:
        axes = axes[np.newaxis, :]

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

        samples = pred.values  # [sample, lat, lon]
        n_show = min(5, samples.shape[0])

        # Ensemble mean for first panel
        ens_mean = np.nanmean(samples, axis=0)
        vmin = np.nanpercentile(samples[:n_show], 2)
        vmax = np.nanpercentile(samples[:n_show], 98)

        im = axes[row, 0].imshow(ens_mean, origin="lower", cmap="cividis",
                                  vmin=vmin, vmax=vmax, aspect="auto")
        label = name.replace("_", "\n")
        metrics_str = (f"rough_lat={metrics['roughness_lat']:.2f}\n"
                       f"rough_lon={metrics['roughness_lon']:.2f}\n"
                       f"spread={metrics['sample_spread']:.2f}")
        axes[row, 0].set_ylabel(f"{label}\n\n{metrics_str}", fontsize=9,
                                 rotation=0, labelpad=100, va="center")
        axes[row, 0].set_title("Ens. Mean" if row == 0 else "", fontsize=10)

        for i in range(n_show):
            ax = axes[row, i + 1]
            ax.imshow(samples[i], origin="lower", cmap="cividis",
                      vmin=vmin, vmax=vmax, aspect="auto")
            if row == 0:
                ax.set_title(f"Sample {i}", fontsize=10)

            # Overlay mask contour if available
            if "obs_mask" in ds:
                mask = ds["obs_mask"]
                if "time" in mask.dims:
                    mask = mask.isel(time=0)
                mask_np = mask.values.astype(float)
                if mask_np.ndim > 2:
                    mask_np = mask_np.reshape(mask_np.shape[-2], mask_np.shape[-1])
                ax.contour(mask_np, levels=[0.5], colors="red",
                           linewidths=0.5, origin="lower")

        # Turn off ticks
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide extra panels
        for i in range(n_show + 1, 6):
            axes[row, i].axis("off")

    fig.suptitle("Conditioning Mode Comparison (first timestep)", fontsize=14, y=1.01)
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(out_dir / f"conditioning_comparison.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison figure saved to {out_dir}/conditioning_comparison.png")


def plot_metrics_bar(results, out_dir):
    """Bar chart comparing metrics across experiments."""
    names = list(results.keys())
    metrics_keys = ["roughness_lat", "roughness_lon", "sample_spread"]
    labels = ["Lat Roughness\n(striping proxy)", "Lon Roughness", "Sample Spread"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(names))

    for i, (key, label) in enumerate(zip(metrics_keys, labels)):
        values = [results[n]["metrics"][key] for n in names]
        bars = axes[i].bar(x, values, color=plt.cm.tab10(x / len(names)))
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([n.replace("_", "\n") for n in names],
                                 fontsize=7, rotation=45, ha="right")
        axes[i].set_title(label, fontsize=11)
        axes[i].grid(axis="y", alpha=0.3)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(out_dir / f"conditioning_metrics.{fmt}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics bar chart saved to {out_dir}/conditioning_metrics.png")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load model + datasets once
    model, dataset, dataset_gen = load_model_and_data(device)

    results = {}

    for name, overrides in EXPERIMENTS.items():
        overrides = copy.deepcopy(overrides)
        generate_kwargs = build_generate_kwargs(overrides)

        pl.seed_everything(42)  # same noise for all experiments

        try:
            ds_all = run_experiment(
                name, model, dataset, dataset_gen, generate_kwargs, device,
            )
            metrics = compute_metrics(ds_all)
            results[name] = {"ds": ds_all, "metrics": metrics}
            print(f"  -> Metrics: {metrics}")
        except Exception as e:
            print(f"  !! FAILED: {e}")
            continue

    # Comparison plots
    plot_comparison(results, OUT_ROOT)
    plot_metrics_bar(results, OUT_ROOT)

    # Save metrics summary
    summary = {name: res["metrics"] for name, res in results.items()}
    with open(OUT_ROOT / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nMetrics summary saved to {OUT_ROOT / 'metrics_summary.json'}")

    # Print table
    print(f"\n{'='*80}")
    print(f"{'Experiment':<25} {'Rough_lat':>10} {'Rough_lon':>10} {'Spread':>10} {'Mean':>10} {'Std':>10}")
    print(f"{'-'*80}")
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<25} {m['roughness_lat']:>10.3f} {m['roughness_lon']:>10.3f} {m['sample_spread']:>10.3f} {m['global_mean']:>10.2f} {m['global_std']:>10.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
