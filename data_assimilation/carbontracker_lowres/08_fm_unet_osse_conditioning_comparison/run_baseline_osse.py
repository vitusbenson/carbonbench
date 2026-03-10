"""Baseline OSSE evaluation using osse_runner for all conditioning methods.

Runs the full CarbonTracker OSSE using the Phase 1 osse_runner infrastructure,
producing standardized metrics_summary.json and all 5 diagnostic plot types.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_baseline_osse.py --mask_source xco2 --mask_pattern random
    CUDA_VISIBLE_DEVICES=0 python run_baseline_osse.py --mask_source 3d --mask_pattern satellite
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
from compare_conditioning_osse import (
    EXPERIMENTS_3D,
    EXPERIMENTS_COLUMN,
    build_generate_kwargs,
    get_ground_truth,
    get_mean_ak_on_l10,
    get_pressure_weights_and_ak,
    lat,
    load_model_and_data,
    lon,
)

from neural_transport.inference.osse_runner import run_osse_comparison

RUN_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Baseline OSSE via osse_runner")
    parser.add_argument(
        "--mask_source",
        type=str,
        default="xco2",
        choices=["3d", "xco2"],
        help="'3d': 3D field masking, 'xco2': synthetic XCO2 column obs",
    )
    parser.add_argument(
        "--mask_pattern",
        type=str,
        default="random",
        choices=["random", "satellite", "checkerboard", "vertical", "horizontal"],
    )
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--obs_fraction", type=float, default=0.3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = RUN_DIR / "singlestep" / "baseline" / f"{args.mask_source}_{args.mask_pattern}"

    # Compute mean AK on l10 levels for column OSSE
    ak_10 = None
    if args.mask_source == "xco2":
        ak_10 = get_mean_ak_on_l10()
        print(f"Mean AK on l10 levels: {ak_10}")

    # Load model + dataset
    model, dataset = load_model_and_data(device)

    nlat_val, nlon_val = len(lat), len(lon)

    # Ground truth
    gt_field = get_ground_truth(dataset, nlat_val, nlon_val)

    # Pressure weights and AK for XCO2 metrics
    pressure_weights, ak_weights = get_pressure_weights_and_ak(dataset, nlat_val, nlon_val, ak_10)

    # Select experiments
    all_experiments = EXPERIMENTS_COLUMN if args.mask_source == "xco2" else EXPERIMENTS_3D
    if args.experiments:
        experiments = {k: v for k, v in all_experiments.items() if k in args.experiments}
        if not experiments:
            print(f"No matching experiments. Available: {list(all_experiments.keys())}")
            sys.exit(1)
    else:
        experiments = all_experiments

    # Build base_generate_kwargs (shared across all experiments)
    base_generate_kwargs = build_generate_kwargs(
        {"_masking": True},
        mask_pattern=args.mask_pattern,
        mask_source=args.mask_source,
        ak_10=ak_10,
    )
    base_generate_kwargs["n_samples"] = args.n_samples
    base_generate_kwargs["obs_fraction"] = args.obs_fraction

    # Build per-experiment overrides (only the method-specific settings)
    experiment_overrides = {}
    for name, exp_config in experiments.items():
        overrides = copy.deepcopy(exp_config)
        masking = overrides.pop("_masking", True)
        if not masking:
            overrides["masking"] = False
        experiment_overrides[name] = overrides

    # Run all experiments via osse_runner
    results = run_osse_comparison(
        model=model,
        dataset=dataset,
        experiments=experiment_overrides,
        base_generate_kwargs=base_generate_kwargs,
        device=device,
        out_dir=str(out_dir),
        nlat=nlat_val,
        nlon=nlon_val,
        lat=lat,
        lon=lon,
        pressure_weights=pressure_weights,
        ak=ak_weights,
        gt_field=gt_field,
        seed=42,
    )

    if not results:
        print("No experiments succeeded!")
        sys.exit(1)

    print(f"\nBaseline OSSE complete. Results saved to: {out_dir}")
    print("  - metrics_summary.json")
    print("  - plots/ (5 diagnostic plot types)")


if __name__ == "__main__":
    main()
