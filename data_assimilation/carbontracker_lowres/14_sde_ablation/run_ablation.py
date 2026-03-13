"""
SDE Posterior Sampling Ablation Experiment (Phase 7).

Sweeps sigma_max, noise_schedule, corrector steps/step_size, step count,
and projection using the best unconditional FM model from experiment 11.

Compares SDE (stochastic posterior sampling) against FlowDPS baseline.

Usage:
    python run_ablation.py
    python run_ablation.py --device cuda
    python run_ablation.py --ablation sigma_max
"""

import argparse
import json
from pathlib import Path

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
from neural_transport.inference.metrics import compute_all_metrics
from neural_transport.training import load_model

torch.set_float32_matmul_precision("high")

DEFAULT_TRAINING_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
DEFAULT_FORECAST_DATA_ROOT = DEFAULT_TRAINING_DATA_ROOT + "/test"

CARBONBENCH_ROOT = Path(__file__).resolve().parent.parent
EXP_11_DIR = CARBONBENCH_ROOT / "11_fm_unet_final"
EXP_09_DIR = CARBONBENCH_ROOT / "09_fm_unet_ot_training"
EXP_01_DIR = CARBONBENCH_ROOT / "01_fm_unet_training_baseline"
EXP_DIR = Path(__file__).resolve().parent

grid = "latlon5.625"
vertical_levels = "l10"
freq = "6h"

nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[vertical_levels]["level"])
lat = LATLON_PROTOTYPE_COORDS[grid]["lat"]
lon = LATLON_PROTOTYPE_COORDS[grid]["lon"]

cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(len(lon), axis=1).reshape(-1, 1)
cos_lat = cos_lat / np.mean(cos_lat)

TARGET_VARS = ["co2massmix"]
METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in TARGET_VARS}


def get_base_generate_kwargs():
    return dict(
        n_samples=100,
        refine_start=1.0,
        avg_over_levels=False,
        masking=True,
        mask_source="test",
        mask_pattern="vertical",
        masking_time=None,
        t_threshold=0.9,
        masking_method="total_column_average_simple",
        analyze_masking=False,
        obs_fraction=0.3,
        noise_pattern=None,
        analyze_noise=False,
        steps=21,
        guidance_scale=1.0,
        sigma_obs=0.1,
        spatial_smoothing_sigma=0.0,
        # SDE defaults
        sampler="sde",
        fresh_noise=True,
        sigma_max=0.3,
        noise_schedule="annealed",
        n_corrector_steps=0,
        corrector_step_size=0.01,
        use_projection=True,
    )


# --- Ablation configurations ---

ABLATION_CONFIGS = {}

# 1. Baselines
ABLATION_CONFIGS["unconditional"] = dict(masking=False, sampler=None)
ABLATION_CONFIGS["best_flowdps"] = dict(sampler="flowdps", sigma_obs=0.1)

# 2. sigma_max sweep (no corrector)
for s in [0.1, 0.3, 0.5, 1.0]:
    ABLATION_CONFIGS[f"sde_smax{s}"] = dict(sigma_max=s)

# 3. Noise schedule comparison (sigma_max=0.3)
for sched in ["annealed", "constant", "cosine"]:
    ABLATION_CONFIGS[f"sde_sched_{sched}"] = dict(sigma_max=0.3, noise_schedule=sched)

# 4. Corrector steps sweep (sigma_max=0.3)
for k in [1, 3, 5]:
    ABLATION_CONFIGS[f"pc_c{k}_smax0.3"] = dict(
        sigma_max=0.3, n_corrector_steps=k, corrector_step_size=0.01,
    )

# 5. Corrector step size sweep (sigma_max=0.3, 3 steps)
for eps in [0.001, 0.005, 0.01, 0.05]:
    ABLATION_CONFIGS[f"pc_c3_eps{eps}"] = dict(
        sigma_max=0.3, n_corrector_steps=3, corrector_step_size=eps,
    )

# 6. No projection (pure SDE)
ABLATION_CONFIGS["sde_noproj_smax0.3"] = dict(sigma_max=0.3, use_projection=False)

# 7. Step count sweep
for st in [21, 51, 101]:
    ABLATION_CONFIGS[f"sde_smax0.3_steps{st}"] = dict(sigma_max=0.3, steps=st)


def load_best_model(device="cuda"):
    """Load best model: try exp 11, then 09, then 01."""
    for exp_dir in [EXP_11_DIR, EXP_09_DIR, EXP_01_DIR]:
        try:
            model = load_model(exp_dir, ckpt="best", device=device)
            print(f"Loaded model from {exp_dir}")
            return model
        except Exception as e:
            print(f"Could not load from {exp_dir}: {e}")
    raise RuntimeError("No model checkpoint found")


def run_single_eval(model, generate_kwargs, data_path_forecast, eval_name, out_dir, device="cuda"):
    """Run evaluation with given generate_kwargs, save metrics."""
    obs_compare_path = f"{DEFAULT_FORECAST_DATA_ROOT}/obs_carbontracker_{grid}_{vertical_levels}_{freq}.zarr"

    eval_dir = out_dir / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Evaluating: {eval_name} ---")
    print(f"  Config overrides: {generate_kwargs}")

    try:
        ds_pred = iterative_generate(
            model,
            data_path_forecast,
            device=device,
            freq="QS",
            generate_kwargs=generate_kwargs,
            num_workers=8,
        )

        ds_obs = xr.open_zarr(obs_compare_path).compute()

        metrics = compute_all_metrics(
            ds_pred=ds_pred,
            ds_obs=ds_obs,
            target_vars=TARGET_VARS,
            weights=METRIC_WEIGHTS,
            nlat=len(lat),
            nlon=len(lon),
        )

        metrics_path = eval_dir / "metrics_summary.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"  Metrics saved to {metrics_path}")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v:.4f}")

        return metrics

    except Exception as e:
        print(f"  ERROR in {eval_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run_ablation(ablation_filter=None, device="cuda", out_dir=None):
    """Run SDE ablation experiments."""
    if out_dir is None:
        out_dir = EXP_DIR / "results"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_best_model(device=device)
    data_path_forecast = Path(DEFAULT_FORECAST_DATA_ROOT)

    if ablation_filter is not None:
        configs = {k: v for k, v in ABLATION_CONFIGS.items() if ablation_filter in k}
        if not configs:
            print(f"No configs matching '{ablation_filter}'. Available: {list(ABLATION_CONFIGS.keys())}")
            return {}
    else:
        configs = ABLATION_CONFIGS

    all_results = {}
    for name, overrides in configs.items():
        gen_kwargs = get_base_generate_kwargs()
        gen_kwargs.update(overrides)
        metrics = run_single_eval(model, gen_kwargs, data_path_forecast, name, out_dir, device)
        all_results[name] = metrics

    # Save combined summary
    summary_path = out_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*120}")
    print(f"{'Config':<30} {'RMSE_3d':>10} {'RMSE_xco2':>12} {'RMSE_xco2_obs':>14} {'Spread_3d':>10} {'SSR':>8}")
    print(f"{'-'*120}")
    for name, metrics in all_results.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            rmse_3d = metrics.get("co2massmix_delta_rmse", float("nan"))
            rmse_xco2 = metrics.get("co2massmix_delta_rmse_xco2", float("nan"))
            rmse_xco2_obs = metrics.get("co2massmix_delta_rmse_xco2_obs", float("nan"))
            spread = metrics.get("co2massmix_delta_spread_3d", float("nan"))
            ssr = metrics.get("co2massmix_delta_spread_skill_ratio", float("nan"))
            print(f"  {name:<28} {rmse_3d:>10.4f} {rmse_xco2:>12.4f} {rmse_xco2_obs:>14.4f} {spread:>10.4f} {ssr:>8.4f}")
        else:
            print(f"  {name:<28} ERROR: {metrics}")
    print(f"{'='*120}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="SDE Posterior Sampling Ablation (Phase 7)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ablation", type=str, default=None, help="Filter configs by substring")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    run_ablation(
        ablation_filter=args.ablation,
        device=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
