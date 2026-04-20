"""Ensemble trajectory evaluation for the Phase 24 FM transport-prior model.

For each random init point, draws N independent samples and rolls each one out
as an auto-regressive (or sliding-window) trajectory. Winds come from GT at
every step; CO2 is fed back from each sample's own prediction.

Usage:
    python eval_autoregressive.py
    python eval_autoregressive.py --reinit-every 120
    python eval_autoregressive.py --n-init-points 5 --n-samples 5 --n-steps 40
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.analyse import compute_trajectory_ensemble_metrics
from neural_transport.inference.generation import generate_ensemble
from neural_transport.training.train import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"

TARGET_VARS = ["co2massmix"]
FORCING_VARS = ["co2massmix", "u", "v"]
# 6h freq → 4 steps/day → 365*4 = 1460 steps per year.
STEPS_PER_YEAR = 365 * 4


def parse_n_steps(val, max_steps):
    if val is None or val == "full":
        return max_steps
    return min(int(val), max_steps)


def main():
    parser = argparse.ArgumentParser(description="Phase 24 ensemble trajectory eval")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--n-init-points", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-steps", type=str, default="full",
                        help="Trajectory length per init. 'full' = 1 year. Capped at 1 year.")
    parser.add_argument("--reinit-every", type=int, default=None,
                        help="Sliding-window re-init frequency in steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    mode_name = f"slidingwindow_{args.reinit_every}" if args.reinit_every else "autoregressive"
    out_dir = EXP_DIR / "singlestep" / "preds" / f"eval_{mode_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(EXP_DIR, ckpt=args.ckpt, device=args.device)
    logger.info("Loaded model (ckpt=%s)", args.ckpt)

    data_cfg = DataConfig(target_vars=TARGET_VARS, forcing_vars=FORCING_VARS)
    loader = InferenceDataLoader(data_cfg, Path(args.data_root) / args.split)

    # Cap trajectory to 1 year; exclude last year (= n_steps) from init sampling
    # so every trajectory fits inside the split.
    n_steps = parse_n_steps(args.n_steps, STEPS_PER_YEAR)
    valid_init_range = len(loader) - n_steps - 1
    if valid_init_range <= 0:
        raise RuntimeError(
            f"Test split too short for n_steps={n_steps} (len={len(loader)})."
        )

    rng = np.random.RandomState(args.seed)
    n_inits = min(args.n_init_points, valid_init_range)
    init_indices = sorted(rng.choice(valid_init_range, n_inits, replace=False).tolist())
    logger.info(
        "Rolling out %d inits × %d samples × %d steps (reinit_every=%s)",
        n_inits, args.n_samples, n_steps, args.reinit_every,
    )

    preds = generate_ensemble(
        model,
        loader,
        init_indices=init_indices,
        n_samples=args.n_samples,
        n_steps=n_steps,
        reinit_every=args.reinit_every,
        target_var=TARGET_VARS[0],
        device=args.device,
        seed=args.seed,
        verbose=True,
    )
    preds_path = out_dir / "preds_ensemble.zarr"
    preds.to_zarr(preds_path, mode="w")
    logger.info("Saved ensemble predictions → %s", preds_path)

    # Build GT aligned to (init, lead): fetch through the dataset so we reuse
    # the same normalization-free raw fields the model sees as co2massmix_next.
    target = TARGET_VARS[0]
    nlat, nlon = loader.grid_info.nlat, loader.grid_info.nlon
    nlev = preds.sizes["level"]
    gt_stack = np.full((len(init_indices), n_steps, nlat, nlon, nlev), np.nan, dtype=np.float32)
    for i, init_idx in enumerate(init_indices):
        for k in range(n_steps):
            sample = loader.dataset[init_idx + k]
            next_key = f"{target}_next"
            field = sample[next_key] if next_key in sample else loader.dataset[init_idx + k + 1][target]
            if hasattr(field, "numpy"):
                field = field.numpy()
            if field.ndim == 3:
                field = field[0]
            gt_stack[i, k] = field.reshape(nlat, nlon, nlev)

    gt_ds = xr.Dataset(
        {TARGET_VARS[0]: (("init", "lead", "lat", "lon", "level"), gt_stack)},
        coords={
            "init": preds["init"].values,
            "lead": preds["lead"].values,
            "lat": preds["lat"].values,
            "lon": preds["lon"].values,
            "level": preds["level"].values,
        },
    )

    per_lead, summary, rank_hist = compute_trajectory_ensemble_metrics(
        gt_ds, preds, target_var=TARGET_VARS[0]
    )
    score_dir = out_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    per_lead.to_csv(score_dir / "metrics_per_lead.csv")
    summary.to_csv(score_dir / "metrics_summary.csv", header=["value"])
    pd.DataFrame(rank_hist).to_csv(score_dir / "rank_histogram.csv", index_label="lead")
    logger.info("Metrics → %s", score_dir)


if __name__ == "__main__":
    main()
