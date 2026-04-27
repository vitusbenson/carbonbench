"""Phase 25c v4 phase 2: AR rollout eval for the residual FM head.

Unconditional ensemble rollout (n_inits=20, n_samples=10, n_steps=120) using
ResidualFlowMatching: residual ODE-solved on top of frozen f_det at every step.
Compares against:
  D1 (Phase 24 FM, deterministic 1-sample)        9.93 ppm
  Phase 24 ensemble mean ρ=1.0                    9.62 ppm
  25e ρ=1.05                                      6.89 ppm
  D-Flow + ρ=1.05                                 4.75 ppm
  v4 phase 1b (deterministic + rollout-FT, 1-sample)  3.81 ppm
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.analyse import compute_trajectory_ensemble_metrics
from neural_transport.inference.generation import generate_trajectory_ensemble_batched
from neural_transport.training.train import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
DEFAULT_GT = (
    "/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/"
    "data_assimilation/carbontracker_lowres/25_transport_prior_osse/"
    "results/none_1month_v2/gt_trajectory.zarr"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--n-inits", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=120)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--tag", default="residual_fm_1month")
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--gt", default=DEFAULT_GT)
    args = p.parse_args()

    model = load_model(EXP_DIR, ckpt=args.ckpt, device=args.device)
    logger.info("Loaded ResidualFlowMatching model")

    data_cfg = DataConfig(
        dataset="carbontracker", grid="latlon5.625", vertical_levels="l10",
        freq="6h",
        target_vars=["co2massmix", "p_bottom", "p_top"],
        forcing_vars=["co2massmix", "u", "v"],
    )
    loader = InferenceDataLoader(data_cfg, data_path=f"{args.data_root}/test")
    loader.load_dataset()

    rng = np.random.RandomState(args.seed)
    valid = len(loader) - args.n_steps - 1
    init_indices = sorted(rng.choice(valid, args.n_inits, replace=False).tolist())
    logger.info("inits=%s n_samples=%d", init_indices, args.n_samples)

    free_kwargs = {"noise_scale": args.noise_scale} if args.noise_scale != 1.0 else None

    t0 = time.perf_counter()
    ds = generate_trajectory_ensemble_batched(
        model, loader,
        init_indices=init_indices,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        sampler_generate_kwargs=None,
        free_generate_kwargs=free_kwargs,
        obs_every=1,
        obs_offset=0,
        device=args.device,
        seed=args.seed,
        chunk_size=args.chunk_size,
        verbose=True,
    )
    wall = time.perf_counter() - t0
    logger.info("Trajectory rollout done in %.1fs", wall)

    out_dir = EXP_DIR / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = out_dir / "preds_trajectory.zarr"
    if pf.exists():
        import shutil; shutil.rmtree(pf)
    ds.to_zarr(pf, mode="w")

    import shutil
    gt_path = out_dir / "gt_trajectory.zarr"
    if gt_path.exists(): shutil.rmtree(gt_path)
    shutil.copytree(args.gt, gt_path)
    gt_ds = xr.open_zarr(gt_path)

    pl_, sm, rh = compute_trajectory_ensemble_metrics(gt_ds, ds, target_var="co2massmix")
    sd = out_dir / "scores"; sd.mkdir(exist_ok=True)
    pl_.to_csv(sd / "metrics_per_lead.csv")
    sm.to_csv(sd / "metrics_summary.csv", header=["value"])
    pd.DataFrame(rh).to_csv(sd / "rank_histogram.csv", index_label="lead")

    logger.info("RESIDUAL_FM RMSE=%.3f CRPS=%.3f spread/err=%.3f",
                float(sm["rmse_mean"]), float(sm["crps"]), float(sm["spread_error_ratio"]))
    logger.info("  RMSE@+6h=%.3f @+7d=%.3f @+30d=%.3f",
                float(pl_["rmse_mean"].iloc[0]),
                float(pl_["rmse_mean"].iloc[27]),
                float(pl_["rmse_mean"].iloc[-1]))


if __name__ == "__main__":
    main()
