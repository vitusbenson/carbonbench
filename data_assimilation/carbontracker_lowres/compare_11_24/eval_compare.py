"""Cross-model ensemble comparison: Phase 11 (unconditional) vs Phase 24 (conditional).

For matched init_indices + seed, runs `generate_ensemble` on both models and
writes per-model ensemble zarrs + a combined per-lead metrics CSV.

Modes:
    --mode onestep     : n_steps=1, 20 inits × 10 samples (~1 min/model).
    --mode trajectory  : n_steps=N per model (default 240 = 60 days @6h).
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

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"

PHASE24_DIR = Path(
    "/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/"
    "data_assimilation/carbontracker_lowres/24_fm_unet_transport_prior"
)
PHASE11_DIR = Path(
    "/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/"
    "data_assimilation/carbontracker_lowres/11_fm_unet_final"
)

TARGET = "co2massmix"
# A shared forcing list works for both: Phase 11 model ignores u/v internally.
SHARED_FORCINGS = ["co2massmix", "u", "v"]


def build_gt(loader, init_indices, n_steps):
    nlat, nlon = loader.grid_info.nlat, loader.grid_info.nlon
    sample0 = loader.dataset[init_indices[0]][TARGET]
    nlev = sample0.shape[-1] if sample0.ndim == 2 else sample0.shape[-1]
    gt = np.full((len(init_indices), n_steps, nlat, nlon, nlev), np.nan, dtype=np.float32)
    for i, init_idx in enumerate(init_indices):
        for k in range(n_steps):
            sample = loader.dataset[init_idx + k]
            next_key = f"{TARGET}_next"
            field = sample[next_key] if next_key in sample else loader.dataset[init_idx + k + 1][TARGET]
            if hasattr(field, "numpy"):
                field = field.numpy()
            if field.ndim == 3:
                field = field[0]
            gt[i, k] = field.reshape(nlat, nlon, nlev)
    return gt


def run_for_model(name, exp_dir, loader, init_indices, n_samples, n_steps, seed, device, out_dir):
    model = load_model(exp_dir, ckpt="best", device=device)
    logger.info("[%s] model loaded from %s", name, exp_dir)
    preds = generate_ensemble(
        model, loader,
        init_indices=init_indices, n_samples=n_samples, n_steps=n_steps,
        target_var=TARGET, device=device, seed=seed, verbose=True,
    )
    preds_path = out_dir / f"preds_{name}.zarr"
    preds.to_zarr(preds_path, mode="w")
    logger.info("[%s] saved preds → %s", name, preds_path)
    # Free GPU memory before loading the next model.
    del model
    import torch
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--mode", choices=["onestep", "trajectory"], default="onestep")
    p.add_argument("--n-init-points", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=None,
                   help="trajectory mode only; default 240 (60 days)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    n_steps = 1 if args.mode == "onestep" else (args.n_steps or 240)
    out_dir = HERE / f"preds_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(target_vars=[TARGET], forcing_vars=SHARED_FORCINGS)
    loader = InferenceDataLoader(data_cfg, Path(args.data_root) / args.split)

    rng = np.random.RandomState(args.seed)
    valid_range = len(loader) - n_steps - 1
    n_inits = min(args.n_init_points, valid_range)
    init_indices = sorted(rng.choice(valid_range, n_inits, replace=False).tolist())
    logger.info("mode=%s n_inits=%d n_samples=%d n_steps=%d seed=%d",
                args.mode, n_inits, args.n_samples, n_steps, args.seed)
    logger.info("init_indices = %s", init_indices)

    # GT once (both models compare against the same GT).
    gt_arr = build_gt(loader, init_indices, n_steps)
    gt_ds = xr.Dataset(
        {TARGET: (("init", "lead", "lat", "lon", "level"), gt_arr)},
        coords={
            "init": np.array(init_indices),
            "lead": np.arange(n_steps),
            "lat": loader.grid_info.lat,
            "lon": loader.grid_info.lon,
            "level": loader.grid_info.levels,
        },
    )
    gt_ds.to_zarr(out_dir / "gt.zarr", mode="w")

    all_rows = []
    for name, exp_dir in [("phase11", PHASE11_DIR), ("phase24", PHASE24_DIR)]:
        preds = run_for_model(
            name, exp_dir, loader, init_indices, args.n_samples, n_steps,
            args.seed, args.device, out_dir,
        )
        per_lead, summary, _ = compute_trajectory_ensemble_metrics(gt_ds, preds, target_var=TARGET)
        per_lead.insert(0, "model", name)
        all_rows.append(per_lead)
        summary.to_csv(out_dir / f"summary_{name}.csv", header=["value"])
        logger.info("[%s] summary:\n%s", name, summary)

    combined = pd.concat(all_rows).reset_index()
    combined.to_csv(out_dir / "metrics_per_lead_combined.csv", index=False)
    logger.info("combined metrics → %s", out_dir / "metrics_per_lead_combined.csv")


if __name__ == "__main__":
    main()
