"""Phase 25b: batched per-trajectory auto-regressive DA.

Identical purpose to ``eval_trajectory.py`` but uses
``generate_trajectory_ensemble_batched`` so that all `n_inits × n_samples`
trajectories are propagated in a single batch per step. This is much faster
on multi-GB-VRAM GPUs and lets us bump `n_inits` to 20 and `n_samples` to 10
within the same wallclock as the old 4×4 setup.

Usage:
    python eval_trajectory_v2.py --method fmps --n-inits 20 --n-samples 10 \
        --n-steps 120 --obs-every 4 --tag 1month_v2

    python eval_trajectory_v2.py --method none --n-inits 20 --n-samples 10 \
        --n-steps 120 --tag 1month_v2

    python eval_trajectory_v2.py --method dflow --n-inits 20 --n-samples 10 \
        --n-steps 120 --obs-every 4 --n-opt-steps 30 --tag 1month_v2
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
    generate_trajectory_enkf,
    generate_trajectory_enks,
    generate_trajectory_ensemble_batched,
    generate_trajectory_window_dflow,
)
from neural_transport.training.train import load_model

from configs import (  # noqa: E402
    adapt_for_trajectory,
    free_kwargs_from_obs_kwargs,
    load_method_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
PHASE24_DIR = EXP_DIR.parent / "24_fm_unet_transport_prior"
PHASE25G_DIR = (
    EXP_DIR.parent / "25c_v4_residual_fm" / "phase2_residual_fm"
)
PHASE25G_ROLLOUT_FT_DIR = (
    EXP_DIR.parent / "25c_v4_residual_fm" / "phase2b_residual_fm_rollout_ft"
)
PHASE25P_DIR = (
    EXP_DIR.parent / "25c_v4_residual_fm" / "phase2c_residual_fm_ema"
)
PHASE25P2_DIR = (
    EXP_DIR.parent / "25c_v4_residual_fm" / "phase2p_residual_fm_stable"
)
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["fmps", "dflow", "none", "window_dflow", "enkf", "enks"], default="fmps")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--n-inits", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=120)
    p.add_argument("--obs-every", type=int, default=4)
    p.add_argument("--obs-offset", type=int, default=0)
    p.add_argument("--reinit-every", type=int, default=None)
    p.add_argument("--n-opt-steps", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None,
                   help="If set, run forward in chunks of this many trajectories.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--tag", default="1month_v2")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="AWG-style initial-noise scaling rho. >1 widens the source "
                        "distribution to counter AR underdispersion (Phase 25e).")
    p.add_argument("--phase", choices=["24", "25g", "25g_rollout_ft", "25p", "25p2"], default="24",
                   help="Which model to load: 24, 25g, 25g_rollout_ft, 25p (Phase 2c residual-FM with EMA + 40k steps), "
                        "or 25p2 (Phase 25p.B residual-FM on EMA-frozen phase1p_uniform f_det).")
    p.add_argument("--ema", action="store_true",
                   help="Load EMA shadow weights from the checkpoint (FM/diffusion best practice).")
    p.add_argument("--model-dir", default=None,
                   help="Override the phase-based model directory (e.g. a leak-free P2 run dir).")
    p.add_argument("--obs-fraction", type=float, default=None,
                   help="Override the satellite-mask obs_fraction (default 0.3 from configs.py).")
    # Phase 25i: per-knob FMPS overrides (post-Optuna sweep).
    p.add_argument("--ode-steps", type=int, default=None,
                   help="Override FM ODE solver steps (default 21).")
    p.add_argument("--guidance-strength", type=float, default=None,
                   help="Override FMPS guidance_strength (default tuned ~46).")
    p.add_argument("--spatial-smoothing", type=float, default=None,
                   help="Override FMPS spatial_smoothing_sigma (default tuned ~4.0).")
    p.add_argument("--grad-clip-norm", type=float, default=None,
                   help="Override FMPS grad_clip_norm (default tuned ~9.9).")
    # window_dflow specific
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--window-stride", type=int, default=None,
                   help="Default = window_size (non-overlapping).")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--sigma-obs", type=float, default=0.1)
    p.add_argument("--reg-weight", type=float, default=0.0)
    p.add_argument("--no-checkpointing", action="store_true",
                   help="Disable per-AR-step gradient checkpointing (uses more memory).")
    # enkf specific
    p.add_argument("--enkf-inflation", type=float, default=1.0,
                   help="Multiplicative ensemble inflation post-EnKF update.")
    p.add_argument("--enkf-loc-sigma", type=float, default=0.0,
                   help="Horizontal Gaussian localization sigma in grid cells (0 = vertical-only EnKF).")
    p.add_argument("--enkf-prior-inflation", type=float, default=1.0,
                   help="Multiplicative prior-inflation factor (Anderson 2007), applied BEFORE the EnKF update.")
    p.add_argument("--enkf-hybrid", action="store_true",
                   help="EnKF + FMPS hybrid: use FMPS sampler at obs steps (instead of free) then apply EnKF on top.")
    p.add_argument("--enkf-global-bias-correct", action="store_true",
                   help="After per-cell EnKF, snap global mean XCO2 to observed mean. Counters residual-FM drift.")
    # enks specific
    p.add_argument("--enks-lag", type=int, default=24,
                   help="Fixed-lag EnKS smoother window (in AR steps). Default 24 (1 day at 6h freq).")
    p.add_argument("--enks-damping", type=float, default=1.0,
                   help="Scale factor for past-state Kalman gain (0=no smoothing, 1=full). Mitigates spurious cross-cov from small ensemble.")
    args = p.parse_args()

    if args.phase == "25p2":
        model_dir = PHASE25P2_DIR
    elif args.phase == "25p":
        model_dir = PHASE25P_DIR
    elif args.phase == "25g_rollout_ft":
        model_dir = PHASE25G_ROLLOUT_FT_DIR
    elif args.phase == "25g":
        model_dir = PHASE25G_DIR
    else:
        model_dir = PHASE24_DIR
    if args.model_dir is not None:
        model_dir = Path(args.model_dir)
    model = load_model(model_dir, ckpt=args.ckpt, device=args.device, ema=args.ema)
    logger.info("Loaded model from %s (phase=%s)", model_dir, args.phase)

    data_cfg = DataConfig(
        dataset="carbontracker",
        grid="latlon5.625",
        vertical_levels="l10",
        freq="6h",
        target_vars=["co2massmix", "p_bottom", "p_top"],
        forcing_vars=["co2massmix", "u", "v"],
    )
    loader = InferenceDataLoader(data_cfg, data_path=f"{args.data_root}/{args.split}")
    loader.load_dataset()

    valid_init_range = len(loader) - args.n_steps - 1
    if valid_init_range <= 0:
        raise RuntimeError(f"Split too short for n_steps={args.n_steps}")
    rng = np.random.RandomState(args.seed)
    init_indices = sorted(rng.choice(valid_init_range, min(args.n_inits, valid_init_range),
                                     replace=False).tolist())
    logger.info("inits=%s n_samples=%d n_steps=%d obs_every=%d method=%s chunk=%s",
                init_indices, args.n_samples, args.n_steps, args.obs_every,
                args.method, args.chunk_size)

    out_dir = EXP_DIR / "results" / f"{args.method}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "none":
        sampler_kwargs = None
        free_kwargs = {"noise_scale": args.noise_scale} if args.noise_scale != 1.0 else None
    elif args.method in ("enkf", "enks"):
        # Reuse FMPS config to inherit mask_pattern/obs_fraction/ak_10/etc.
        cfg = load_method_config("fmps", n_samples=args.n_samples)
        sampler_kwargs = adapt_for_trajectory(
            cfg, n_samples=args.n_samples, method="fmps", n_opt_steps=None,
        )
        if args.noise_scale != 1.0:
            sampler_kwargs["noise_scale"] = args.noise_scale
        if args.obs_fraction is not None:
            sampler_kwargs["obs_fraction"] = args.obs_fraction
        free_kwargs = free_kwargs_from_obs_kwargs(sampler_kwargs)
    elif args.method == "window_dflow":
        # Reuse the FMPS config to inherit mask_pattern / obs_fraction / etc.,
        # but window-D-Flow uses its own loss machinery (no inner sampler).
        cfg = load_method_config("fmps", n_samples=args.n_samples)
        sampler_kwargs = adapt_for_trajectory(
            cfg, n_samples=args.n_samples, method="fmps", n_opt_steps=None,
        )
        if args.noise_scale != 1.0:
            sampler_kwargs["noise_scale"] = args.noise_scale
        if args.obs_fraction is not None:
            sampler_kwargs["obs_fraction"] = args.obs_fraction
        free_kwargs = free_kwargs_from_obs_kwargs(sampler_kwargs)
    else:
        cfg = load_method_config(args.method, n_samples=args.n_samples)
        sampler_kwargs = adapt_for_trajectory(
            cfg, n_samples=args.n_samples, method=args.method,
            n_opt_steps=args.n_opt_steps,
        )
        if args.noise_scale != 1.0:
            sampler_kwargs["noise_scale"] = args.noise_scale
        if args.obs_fraction is not None:
            sampler_kwargs["obs_fraction"] = args.obs_fraction
        # Phase 25i FMPS knob overrides
        if args.ode_steps is not None:
            sampler_kwargs["steps"] = args.ode_steps
        if args.guidance_strength is not None:
            sampler_kwargs["guidance_strength"] = args.guidance_strength
        if args.spatial_smoothing is not None:
            sampler_kwargs["spatial_smoothing_sigma"] = args.spatial_smoothing
        if args.grad_clip_norm is not None:
            sampler_kwargs["grad_clip_norm"] = args.grad_clip_norm
        free_kwargs = free_kwargs_from_obs_kwargs(sampler_kwargs)

    t0 = time.perf_counter()
    if args.method == "enkf":
        ds = generate_trajectory_enkf(
            model, loader,
            init_indices=init_indices,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            obs_kwargs=sampler_kwargs,
            free_kwargs=free_kwargs,
            sampler_kwargs=(sampler_kwargs if args.enkf_hybrid else None),
            obs_every=args.obs_every,
            obs_offset=args.obs_offset,
            sigma_obs=args.sigma_obs,
            inflation=args.enkf_inflation,
            prior_inflation=args.enkf_prior_inflation,
            loc_sigma=args.enkf_loc_sigma,
            global_bias_correct=args.enkf_global_bias_correct,
            device=args.device,
            seed=args.seed,
            chunk_size=args.chunk_size,
            verbose=True,
        )
    elif args.method == "enks":
        ds = generate_trajectory_enks(
            model, loader,
            init_indices=init_indices,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            obs_kwargs=sampler_kwargs,
            free_kwargs=free_kwargs,
            obs_every=args.obs_every,
            obs_offset=args.obs_offset,
            sigma_obs=args.sigma_obs,
            inflation=args.enkf_inflation,
            prior_inflation=args.enkf_prior_inflation,
            loc_sigma=args.enkf_loc_sigma,
            lag=args.enks_lag,
            damping=args.enks_damping,
            device=args.device,
            seed=args.seed,
            chunk_size=args.chunk_size,
            verbose=True,
        )
    elif args.method == "window_dflow":
        ds = generate_trajectory_window_dflow(
            model, loader,
            init_indices=init_indices,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            obs_kwargs=sampler_kwargs,
            free_kwargs=free_kwargs,
            obs_every=args.obs_every,
            obs_offset=args.obs_offset,
            window_size=args.window_size,
            window_stride=args.window_stride,
            n_opt_steps=args.n_opt_steps if args.n_opt_steps is not None else 20,
            lr=args.lr,
            sigma_obs=args.sigma_obs,
            reg_weight=args.reg_weight,
            device=args.device,
            seed=args.seed,
            chunk_size=args.chunk_size,
            use_checkpointing=not args.no_checkpointing,
            verbose=True,
        )
    else:
        ds = generate_trajectory_ensemble_batched(
            model, loader,
            init_indices=init_indices,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            sampler_generate_kwargs=sampler_kwargs,
            free_generate_kwargs=free_kwargs,
            obs_every=args.obs_every,
            obs_offset=args.obs_offset,
            reinit_every=args.reinit_every,
            device=args.device,
            seed=args.seed,
            chunk_size=args.chunk_size,
            verbose=True,
        )
    wall = time.perf_counter() - t0
    logger.info("Trajectory rollout done in %.1fs", wall)

    preds_path = out_dir / "preds_trajectory.zarr"
    if preds_path.exists():
        import shutil; shutil.rmtree(preds_path)
    ds.to_zarr(preds_path, mode="w")
    logger.info("Saved → %s", preds_path)

    # Build GT.
    target = "co2massmix"
    nlat, nlon = loader.grid_info.nlat, loader.grid_info.nlon
    nlev = ds.sizes["level"]
    gt_stack = np.full((len(init_indices), args.n_steps, nlat, nlon, nlev),
                       np.nan, dtype=np.float32)
    ds_inner = loader.dataset
    fast_arr = getattr(ds_inner, "_fast_var_data", {}).get(target)
    if fast_arr is not None:
        init_offset = ds_inner.initial_time_idx
        for i, init_idx in enumerate(init_indices):
            start = init_offset + init_idx + 1
            end = min(start + args.n_steps, fast_arr.shape[0])
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
    gt_path = out_dir / "gt_trajectory.zarr"
    if gt_path.exists():
        import shutil; shutil.rmtree(gt_path)
    gt_ds.to_zarr(gt_path, mode="w")

    per_lead, summary, rank_hist = compute_trajectory_ensemble_metrics(gt_ds, ds, target_var=target)
    score_dir = out_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    per_lead.to_csv(score_dir / "metrics_per_lead.csv")
    summary.to_csv(score_dir / "metrics_summary.csv", header=["value"])
    pd.DataFrame(rank_hist).to_csv(score_dir / "rank_histogram.csv", index_label="lead")

    info = {
        "eval": f"trajectory_v2_{args.tag}",
        "phase": args.phase,
        "method": args.method,
        "wall_time_sec": wall,
        "n_inits": len(init_indices),
        "n_samples_per_init": args.n_samples,
        "n_steps": args.n_steps,
        "obs_every": args.obs_every,
        "obs_offset": args.obs_offset,
        "init_indices": init_indices,
        "ckpt": args.ckpt,
        "chunk_size": args.chunk_size,
        "n_opt_steps": args.n_opt_steps,
    }
    with open(out_dir / "method_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    logger.info("Metrics → %s | summary RMSE=%.3f CRPS=%.3f spread/err=%.2f",
                score_dir, float(summary["rmse_mean"]), float(summary["crps"]),
                float(summary["spread_error_ratio"]))


if __name__ == "__main__":
    main()
