"""P3: MIP-style OSSE — flow-matching DA on MIP-realistic synthetic XCO2.

Unlike the idealised OSSE (``25_transport_prior_osse/eval_trajectory_v2.py``,
which uses synthetic swath masks + a uniform averaging kernel), this runner
samples observations at the *real* OCO-2 orbit footprints with the *real*
20-level retrieval averaging kernels, synthesised from a known CarbonTracker
truth through the corrected (interpolate-then-apply) P1 forward operator. It
then runs the leak-free (P2) residual-FM EnKF and scores the reconstruction
against the known truth.

The XCO2 *values* are synthetic (truth → P1 operator → +retrieval noise); only
the sampling geometry + averaging kernels come from the staged MIP product.

Usage (smoke):
    python eval_mip_osse.py --method enkf --n-inits 4 --n-samples 6 \
        --n-steps 24 --obs-every 4 --enkf-loc-sigma 4 --sigma-obs 0.1 \
        --tag smoke

    python eval_mip_osse.py --method none --n-inits 4 --n-samples 6 \
        --n-steps 24 --tag smoke   # free (no-DA) baseline
"""

import argparse
import json
import logging
import sys
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
    generate_trajectory_ensemble_batched,
)
from neural_transport.inference.orbit_obs import OrbitObsProvider
from neural_transport.training.train import load_model

# Reuse the idealised-OSSE FMPS/D-Flow config helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "25_transport_prior_osse"))
from configs import adapt_for_trajectory, load_method_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent / "25c_v4_residual_fm"
DEFAULT_MODEL_DIR = ROOT / "phase2p_residual_fm_stable_leakfree"
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker_leakfree"
# MIP OCO-2 l20 product whose "train" split spans the 2014-2020 MIP period.
DEFAULT_ORBIT_ZARR = (
    "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/OCO2MIP_OCO2/train/"
    "mip_oco2_latlon5.625_l20_6h.zarr"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["enkf", "fmps", "none"], default="enkf",
                   help="enkf, none (free), or fmps (= any posterior sampler; see --sampler).")
    p.add_argument("--sampler", default="fmps",
                   choices=["fmps", "flowdps", "sde", "mcg", "pcfm", "dps", "fig", "ictm"],
                   help="Posterior sampler used when --method fmps. flowdps = closed-form "
                        "PGDM projection (best for our linear operator).")
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--orbit-zarr", default=DEFAULT_ORBIT_ZARR)
    p.add_argument("--n-inits", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=120)
    p.add_argument("--obs-every", type=int, default=4)
    p.add_argument("--obs-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--tag", default="mip_osse")
    p.add_argument("--noise-scale", type=float, default=1.05,
                   help="Initial-noise scaling rho for the residual-FM source dist.")
    p.add_argument("--sigma-obs", type=float, default=0.1,
                   help="Filter's assumed obs-error std (physical co2massmix).")
    p.add_argument("--enkf-loc-sigma", type=float, default=4.0,
                   help="Horizontal Gaussian localization sigma in grid cells.")
    p.add_argument("--enkf-inflation", type=float, default=1.0)
    p.add_argument("--enkf-prior-inflation", type=float, default=1.0)
    # MIP-realism knobs (P3.4 sweep)
    p.add_argument("--obs-noise", type=float, default=0.0,
                   help="Std of additive retrieval noise on synthetic orbit obs.")
    p.add_argument("--ak-mode", choices=["real", "uniform"], default="real",
                   help="Use the real 20-level AK shape, or flatten it (AK ablation).")
    p.add_argument("--thin-fraction", type=float, default=1.0,
                   help="Keep this fraction of real observed cells (sparsity ablation).")
    # FMPS knob overrides (idealised-tuned defaults transfer poorly to sparse orbits)
    p.add_argument("--spatial-smoothing", type=float, default=None,
                   help="Override FMPS spatial_smoothing_sigma (default ~4.0; too large for sparse orbits).")
    p.add_argument("--guidance-strength", type=float, default=None,
                   help="Override FMPS guidance_strength (default ~46).")
    p.add_argument("--chunk-size", type=int, default=None)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    model = load_model(model_dir, ckpt=args.ckpt, device=args.device)
    logger.info("Loaded leak-free model from %s", model_dir)

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

    nlat, nlon = loader.grid_info.nlat, loader.grid_info.nlon

    orbit_obs = None
    if args.method in ("enkf", "fmps"):
        orbit_obs = OrbitObsProvider(args.orbit_zarr, nlat=nlat, nlon=nlon)
        # Report realised obs coverage over the init windows for transparency.
        times = loader.dataset.ds.time.values
        sample_ts = []
        for i in init_indices:
            for k in range(0, args.n_steps, args.obs_every):
                j = i + k + 1
                if j < len(times):
                    sample_ts.append(times[j])
        frac = orbit_obs.mean_obs_fraction(sample_ts)
        logger.info("orbit obs mean coverage over obs steps: %.3f%% of cells", 100 * frac)

    logger.info("inits=%s n_samples=%d n_steps=%d obs_every=%d method=%s ak=%s thin=%.2f noise=%.3f",
                init_indices, args.n_samples, args.n_steps, args.obs_every,
                args.method, args.ak_mode, args.thin_fraction, args.obs_noise)

    out_dir = EXP_DIR / "results" / f"{args.method}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.method == "enkf":
        ds = generate_trajectory_enkf(
            model, loader,
            init_indices=init_indices,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            obs_kwargs={},
            free_kwargs={"n_samples": 1, "masking": False, "noise_scale": args.noise_scale},
            sampler_kwargs=None,
            obs_every=args.obs_every,
            obs_offset=args.obs_offset,
            sigma_obs=args.sigma_obs,
            inflation=args.enkf_inflation,
            prior_inflation=args.enkf_prior_inflation,
            loc_sigma=args.enkf_loc_sigma,
            orbit_obs=orbit_obs,
            obs_noise=args.obs_noise,
            ak_mode=args.ak_mode,
            thin_fraction=args.thin_fraction,
            device=args.device,
            seed=args.seed,
            chunk_size=args.chunk_size,
            verbose=True,
        )
    elif args.method == "fmps":
        cfg = load_method_config(args.sampler, n_samples=args.n_samples)
        sampler_kwargs = adapt_for_trajectory(cfg, n_samples=args.n_samples, method=args.sampler)
        if args.noise_scale != 1.0:
            sampler_kwargs["noise_scale"] = args.noise_scale
        sampler_kwargs["sigma_obs"] = args.sigma_obs
        if args.spatial_smoothing is not None:
            sampler_kwargs["spatial_smoothing_sigma"] = args.spatial_smoothing
        if args.guidance_strength is not None:
            sampler_kwargs["guidance_strength"] = args.guidance_strength
        ds = generate_trajectory_ensemble_batched(
            model, loader,
            init_indices=init_indices,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            sampler_generate_kwargs=sampler_kwargs,
            free_generate_kwargs={"n_samples": 1, "masking": False, "noise_scale": args.noise_scale},
            obs_every=args.obs_every,
            obs_offset=args.obs_offset,
            orbit_obs=orbit_obs,
            obs_noise=args.obs_noise,
            ak_mode=args.ak_mode,
            thin_fraction=args.thin_fraction,
            device=args.device,
            seed=args.seed,
            chunk_size=args.chunk_size,
            verbose=True,
        )
    else:  # free (no-DA) baseline
        ds = generate_trajectory_ensemble_batched(
            model, loader,
            init_indices=init_indices,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            sampler_generate_kwargs=None,
            free_generate_kwargs={"n_samples": 1, "masking": False, "noise_scale": args.noise_scale},
            obs_every=args.obs_every,
            obs_offset=args.obs_offset,
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

    # Build GT (truth) field for scoring.
    target = "co2massmix"
    nlev = ds.sizes["level"]
    gt_stack = np.full((len(init_indices), args.n_steps, nlat, nlon, nlev), np.nan, dtype=np.float32)
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
            "init": ds["init"].values, "sample": ds["sample"].values if "sample" in ds.coords else [0],
            "lead": ds["lead"].values, "lat": ds["lat"].values,
            "lon": ds["lon"].values, "level": ds["level"].values,
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
        "eval": f"mip_osse_{args.tag}", "method": args.method,
        "model_dir": str(model_dir), "data_root": args.data_root, "split": args.split,
        "orbit_zarr": args.orbit_zarr if args.method == "enkf" else None,
        "wall_time_sec": wall, "n_inits": len(init_indices),
        "n_samples": args.n_samples, "n_steps": args.n_steps,
        "obs_every": args.obs_every, "sigma_obs": args.sigma_obs,
        "loc_sigma": args.enkf_loc_sigma, "obs_noise": args.obs_noise,
        "ak_mode": args.ak_mode, "thin_fraction": args.thin_fraction,
        "init_indices": init_indices,
    }
    with open(out_dir / "method_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    logger.info("Metrics → %s | RMSE=%.3f CRPS=%.3f spread/err=%.2f",
                score_dir, float(summary["rmse_mean"]), float(summary["crps"]),
                float(summary["spread_error_ratio"]))


if __name__ == "__main__":
    main()
