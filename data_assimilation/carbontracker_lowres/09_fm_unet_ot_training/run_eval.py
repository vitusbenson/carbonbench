"""
Evaluation and ablation script for OT-coupled FM training.

Ablations:
  - ot:       Compare baseline (exp 01) vs OT-coupled (exp 09)
  - solver:   Compare midpoint vs dopri5
  - steps:    Compare steps = 11, 21, 51
  - timegrid: Compare uniform vs cosine vs front_loaded
  - all:      Run all ablations

Usage:
    python run_eval.py --ablation all
    python run_eval.py --ablation ot
    python run_eval.py --ablation solver,steps
"""

import argparse
import json
import sys
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
EXP_01_DIR = CARBONBENCH_ROOT / "01_fm_unet_training_baseline"
EXP_09_DIR = Path(__file__).resolve().parent

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
        masking=False,
        mask_source="test",
        mask_pattern="vertical",
        masking_time=None,
        t_threshold=0.9,
        masking_method="interpolate",
        analyze_masking=False,
        obs_fraction=0.3,
        noise_pattern=None,
        analyze_noise=False,
        steps=11,
    )


def load_checkpoint(exp_dir, ckpt="best", device="cuda"):
    """Load a trained model from experiment directory."""
    return load_model(exp_dir, ckpt=ckpt, device=device)


def run_single_eval(model, generate_kwargs, data_path_forecast, eval_name, out_dir, device="cuda"):
    """Run evaluation with given generate_kwargs, save metrics."""
    obs_compare_path = f"{DEFAULT_FORECAST_DATA_ROOT}/obs_carbontracker_{grid}_{vertical_levels}_{freq}.zarr"

    eval_dir = out_dir / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Evaluating: {eval_name} ---")
    print(f"  generate_kwargs: { {k: v for k, v in generate_kwargs.items() if k not in ('data_path_generate',)} }")

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


def ablation_ot(out_dir, device="cuda"):
    """Compare baseline (exp 01) vs OT-coupled (exp 09)."""
    results = {}
    data_path_forecast = Path(DEFAULT_FORECAST_DATA_ROOT)
    gen_kwargs = get_base_generate_kwargs()

    for name, exp_dir in [("baseline_01", EXP_01_DIR), ("ot_09", EXP_09_DIR)]:
        print(f"\nLoading model from {exp_dir}...")
        try:
            model = load_checkpoint(exp_dir, device=device)
            metrics = run_single_eval(model, gen_kwargs, data_path_forecast, name, out_dir, device)
            results[name] = metrics
        except Exception as e:
            print(f"  Could not load {name}: {e}")
            results[name] = {"error": str(e)}

    return results


def ablation_solver(out_dir, device="cuda"):
    """Compare midpoint vs dopri5 solvers."""
    results = {}
    data_path_forecast = Path(DEFAULT_FORECAST_DATA_ROOT)

    # Use exp 09 (OT) model, or fall back to exp 01
    for exp_dir in [EXP_09_DIR, EXP_01_DIR]:
        try:
            model = load_checkpoint(exp_dir, device=device)
            print(f"Using model from {exp_dir}")
            break
        except Exception:
            continue
    else:
        print("ERROR: No model found for solver ablation")
        return {}

    for method in ["midpoint", "dopri5"]:
        gen_kwargs = get_base_generate_kwargs()
        gen_kwargs["method"] = method
        name = f"solver_{method}"
        metrics = run_single_eval(model, gen_kwargs, data_path_forecast, name, out_dir, device)
        results[name] = metrics

    return results


def ablation_steps(out_dir, device="cuda"):
    """Compare different numbers of integration steps."""
    results = {}
    data_path_forecast = Path(DEFAULT_FORECAST_DATA_ROOT)

    for exp_dir in [EXP_09_DIR, EXP_01_DIR]:
        try:
            model = load_checkpoint(exp_dir, device=device)
            print(f"Using model from {exp_dir}")
            break
        except Exception:
            continue
    else:
        print("ERROR: No model found for steps ablation")
        return {}

    for steps in [11, 21, 51]:
        gen_kwargs = get_base_generate_kwargs()
        gen_kwargs["steps"] = steps
        name = f"steps_{steps}"
        metrics = run_single_eval(model, gen_kwargs, data_path_forecast, name, out_dir, device)
        results[name] = metrics

    return results


def ablation_timegrid(out_dir, device="cuda"):
    """Compare uniform vs cosine vs front_loaded time grids."""
    results = {}
    data_path_forecast = Path(DEFAULT_FORECAST_DATA_ROOT)

    for exp_dir in [EXP_09_DIR, EXP_01_DIR]:
        try:
            model = load_checkpoint(exp_dir, device=device)
            print(f"Using model from {exp_dir}")
            break
        except Exception:
            continue
    else:
        print("ERROR: No model found for timegrid ablation")
        return {}

    for spacing in ["uniform", "cosine", "front_loaded"]:
        gen_kwargs = get_base_generate_kwargs()
        gen_kwargs["time_grid_spacing"] = spacing
        name = f"timegrid_{spacing}"
        metrics = run_single_eval(model, gen_kwargs, data_path_forecast, name, out_dir, device)
        results[name] = metrics

    return results


ABLATIONS = {
    "ot": ablation_ot,
    "solver": ablation_solver,
    "steps": ablation_steps,
    "timegrid": ablation_timegrid,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluation + ablation for OT FM training")
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        help="Comma-separated ablation names: ot, solver, steps, timegrid, all",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = EXP_09_DIR / "singlestep" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation == "all":
        ablation_names = list(ABLATIONS.keys())
    else:
        ablation_names = [a.strip() for a in args.ablation.split(",")]

    all_results = {}
    for name in ablation_names:
        if name not in ABLATIONS:
            print(f"Unknown ablation: {name}. Available: {list(ABLATIONS.keys())}")
            continue
        print(f"\n{'='*60}")
        print(f"Running ablation: {name}")
        print(f"{'='*60}")
        results = ABLATIONS[name](out_dir, device=args.device)
        all_results[name] = results

    # Save combined summary
    summary_path = out_dir / "all_ablations_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for ablation_name, results in all_results.items():
        print(f"\n--- {ablation_name} ---")
        for config_name, metrics in results.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                rmse_vals = {k: v for k, v in metrics.items() if "rmse" in k.lower() and isinstance(v, (int, float))}
                rmse_str = ", ".join(f"{k}={v:.4f}" for k, v in rmse_vals.items())
                print(f"  {config_name}: {rmse_str}")
            else:
                print(f"  {config_name}: {metrics}")


if __name__ == "__main__":
    main()
