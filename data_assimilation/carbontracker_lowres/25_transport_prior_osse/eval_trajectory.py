"""Phase 25 Eval 2/3: multi-step auto-regressive posterior conditioning.

For each init, rolls out an ensemble forward by ``n_steps`` leads. At
observation steps, the posterior sampler (FMPS or D-Flow) conditions on a
synthetic XCO2 column mask derived from the GT next field. At unobserved
steps the rollout is unconditional auto-regressive.

GPU memory: D-Flow's autograd graph is bounded per-step (one ODE solve);
since rolled CO2 is detached between steps, trajectory length does not grow
the graph. We still expose ``n-samples``, ``n-opt-steps`` and reinforce
``use_checkpointing`` so long rollouts fit on a single A40.

Usage:
    # 1-month, FMPS, obs every 1 day (=4 steps @6h):
    python eval_trajectory.py --method fmps --n-steps 120 --obs-every 4 \
        --n-inits 4 --n-samples 4 --tag 1month

    # Full test period, FMPS, obs every 4 days:
    python eval_trajectory.py --method fmps --n-steps 1460 --obs-every 16 \
        --n-inits 2 --n-samples 2 --tag full

    # Free auto-regressive baseline:
    python eval_trajectory.py --method none --n-steps 120 --n-inits 4 \
        --n-samples 4 --tag 1month
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.analyse import compute_trajectory_ensemble_metrics
from neural_transport.inference.generation import (
    generate_ensemble,
    generate_trajectory_ensemble_with_obs,
)
from neural_transport.training.train import load_model

from configs import (  # noqa: E402
    adapt_for_trajectory,
    base_generate_config,
    free_kwargs_from_obs_kwargs,
    load_method_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
PHASE24_DIR = EXP_DIR.parent / "24_fm_unet_transport_prior"
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["fmps", "dflow", "none"], default="fmps",
                   help="'none' = unconditional auto-regressive baseline.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--n-inits", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=120)
    p.add_argument("--obs-every", type=int, default=4)
    p.add_argument("--obs-offset", type=int, default=0)
    p.add_argument("--reinit-every", type=int, default=None)
    p.add_argument("--n-opt-steps", type=int, default=None,
                   help="Override D-Flow n_opt_steps for memory/time budgeting.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--tag", default="traj")
    args = p.parse_args()

    model = load_model(PHASE24_DIR, ckpt=args.ckpt, device=args.device)
    logger.info("Loaded Phase 24 model")

    data_cfg = DataConfig(
        dataset="carbontracker",
        grid="latlon5.625",
        vertical_levels="l10",
        freq="6h",
        target_vars=["co2massmix", "p_bottom", "p_top"],
        forcing_vars=["co2massmix", "u", "v"],
    )
    loader = InferenceDataLoader(data_cfg, data_path=f"{args.data_root}/{args.split}")
    _ = loader.load_dataset()

    valid_init_range = len(loader) - args.n_steps - 1
    if valid_init_range <= 0:
        raise RuntimeError(f"Split too short for n_steps={args.n_steps} (len={len(loader)}).")
    rng = np.random.RandomState(args.seed)
    init_indices = sorted(rng.choice(valid_init_range, min(args.n_inits, valid_init_range),
                                     replace=False).tolist())
    logger.info("inits=%s n_samples=%d n_steps=%d obs_every=%d method=%s",
                init_indices, args.n_samples, args.n_steps, args.obs_every, args.method)

    out_dir = EXP_DIR / "results" / f"{args.method}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.method == "none":
        ds = generate_ensemble(
            model, loader,
            init_indices=init_indices,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            reinit_every=args.reinit_every,
            device=args.device,
            seed=args.seed,
            verbose=True,
        )
        ds["obs_present"] = (("lead",), np.zeros(args.n_steps, dtype=bool))
    else:
        cfg = load_method_config(args.method, n_samples=args.n_samples)
        obs_kwargs = adapt_for_trajectory(
            cfg, n_samples=args.n_samples, method=args.method,
            n_opt_steps=args.n_opt_steps,
        )
        free_kwargs = free_kwargs_from_obs_kwargs(obs_kwargs)
        ds = generate_trajectory_ensemble_with_obs(
            model, loader,
            init_indices=init_indices,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            sampler_generate_kwargs=obs_kwargs,
            free_generate_kwargs=free_kwargs,
            obs_every=args.obs_every,
            obs_offset=args.obs_offset,
            reinit_every=args.reinit_every,
            device=args.device,
            seed=args.seed,
            verbose=True,
        )
    wall = time.perf_counter() - t0

    preds_path = out_dir / "preds_trajectory.zarr"
    if preds_path.exists():
        import shutil
        shutil.rmtree(preds_path)
    ds.to_zarr(preds_path, mode="w")
    logger.info("Saved trajectory ensemble → %s (%.1fs)", preds_path, wall)

    # Build GT aligned to (init, lead).
    target = "co2massmix"
    nlat, nlon = loader.grid_info.nlat, loader.grid_info.nlon
    nlev = ds.sizes["level"]
    gt_stack = np.full((len(init_indices), args.n_steps, nlat, nlon, nlev), np.nan, dtype=np.float32)
    ds_inner = loader.dataset
    fast_arr = getattr(ds_inner, "_fast_var_data", {}).get(target)
    if fast_arr is not None:
        # Fast path: slice the precomputed numpy array directly.
        init_offset = ds_inner.initial_time_idx
        for i, init_idx in enumerate(init_indices):
            # target_next at lead k  <=>  target at t+1 where t = init_idx + k
            start = init_offset + init_idx + 1
            end = start + args.n_steps
            end = min(end, fast_arr.shape[0])
            L = end - start
            gt_stack[i, :L] = fast_arr[start:end].reshape(L, nlat, nlon, nlev)
    else:
        for i, init_idx in enumerate(init_indices):
            for k in range(args.n_steps):
                sample = ds_inner[init_idx + k]
                next_key = f"{target}_next"
                field = sample[next_key] if next_key in sample else ds_inner[init_idx + k + 1][target]
                if hasattr(field, "numpy"):
                    field = field.numpy()
                if field.ndim == 3:
                    field = field[0]
                gt_stack[i, k] = field.reshape(nlat, nlon, nlev)

    gt_ds = xr.Dataset(
        {target: (("init", "lead", "lat", "lon", "level"), gt_stack)},
        coords={
            "init": ds["init"].values,
            "lead": ds["lead"].values,
            "lat": ds["lat"].values,
            "lon": ds["lon"].values,
            "level": ds["level"].values,
        },
    )

    per_lead, summary, rank_hist = compute_trajectory_ensemble_metrics(gt_ds, ds, target_var=target)
    score_dir = out_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    per_lead.to_csv(score_dir / "metrics_per_lead.csv")
    summary.to_csv(score_dir / "metrics_summary.csv", header=["value"])
    pd.DataFrame(rank_hist).to_csv(score_dir / "rank_histogram.csv", index_label="lead")

    gt_path = out_dir / "gt_trajectory.zarr"
    if gt_path.exists():
        import shutil
        shutil.rmtree(gt_path)
    gt_ds.to_zarr(gt_path, mode="w")

    info = {
        "eval": f"trajectory_{args.tag}",
        "phase": 25,
        "method": args.method,
        "wall_time_sec": wall,
        "n_inits": len(init_indices),
        "n_samples_per_init": args.n_samples,
        "n_steps": args.n_steps,
        "obs_every": args.obs_every,
        "obs_offset": args.obs_offset,
        "reinit_every": args.reinit_every,
        "init_indices": init_indices,
        "ckpt": args.ckpt,
    }
    with open(out_dir / "method_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    logger.info("Metrics → %s | summary RMSE=%.3f CRPS=%.3f spread/err=%.2f",
                score_dir, float(summary["rmse_mean"]), float(summary["crps"]),
                float(summary["spread_error_ratio"]))


if __name__ == "__main__":
    main()
