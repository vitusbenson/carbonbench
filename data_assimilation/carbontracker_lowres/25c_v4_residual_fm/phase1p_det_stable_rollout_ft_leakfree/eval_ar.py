"""Phase 25c v4 phase 1: AR rollout eval for the deterministic backbone.

Calls the trained UNet directly per step (no FM ODE solve). Compares to:
  D1 (Phase 24 FM, deterministic 1-sample)  9.93 ppm
  Phase 24 ensemble mean ρ=1.0              9.62 ppm
  25e ρ=1.05                                6.89 ppm
  D-Flow + ρ=1.05                           4.75 ppm
"""

import argparse, json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xarray as xr

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.analyse import compute_trajectory_ensemble_metrics
from neural_transport.training.train import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--n-inits", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--ckpt-path", default=None,
                   help="Override: load from explicit ckpt path.")
    p.add_argument("--ema", action="store_true",
                   help="Load EMA shadow weights from the checkpoint.")
    p.add_argument("--tag", default="det_1month_v2")
    args = p.parse_args()

    if args.ckpt_path is not None:
        from neural_transport.litmodule import NeuralTransport
        import torch as _torch
        model = NeuralTransport.load_from_checkpoint(args.ckpt_path, map_location=args.device, weights_only=False)
        if args.ema:
            blob = _torch.load(args.ckpt_path, map_location=args.device, weights_only=False)
            ema_sd = blob.get("ema_state_dict")
            if ema_sd is None:
                logger.warning("ema=True but no ema_state_dict — using raw weights")
            else:
                model.load_state_dict(ema_sd)
                logger.info("Loaded EMA weights from %s", args.ckpt_path)
        model = model.to(args.device).eval()
        logger.info("Loaded model from %s", args.ckpt_path)
    else:
        model = load_model(EXP_DIR, ckpt=args.ckpt, device=args.device, ema=args.ema)
    model.eval()
    inner = getattr(model, "model", model)
    target = "co2massmix"
    logger.info("Loaded deterministic backbone (val loss in name)")

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
    logger.info("inits=%s", init_indices)

    nlat, nlon = loader.grid_info.nlat, loader.grid_info.nlon
    init_batches = [loader.get_batch(i, device=args.device) for i in init_indices]
    sample0 = init_batches[0][target]
    # InferenceDataLoader yields [1, T=1, N, C]
    if sample0.ndim == 4:
        _, _, N, C = sample0.shape
    else:
        _, N, C = sample0.shape
    nlev = C

    # init_batches has tensors of shape [1, T=1, N, C] — squeeze T to get [1, N, C]
    current = torch.cat([b[target].squeeze(1) for b in init_batches], dim=0)  # [n_inits, N, C]

    all_preds = np.full((args.n_inits, 1, args.n_steps, nlat, nlon, C), np.nan, dtype=np.float32)
    times_axis = loader.dataset.ds.time.values
    time_coord = np.empty((args.n_inits, args.n_steps), dtype=times_axis.dtype)

    t0 = time.perf_counter()
    with torch.no_grad():
        for k in range(args.n_steps):
            batches = [loader.get_batch(init_indices[i] + k, device=args.device) for i in range(args.n_inits)]
            batch = {}
            for key, v in batches[0].items():
                if isinstance(v, torch.Tensor):
                    batch[key] = torch.cat([bb[key] for bb in batches], dim=0)
                else:
                    batch[key] = v
            for i, init_idx in enumerate(init_indices):
                time_coord[i, k] = times_axis[init_idx + k + 1]
            batch[target] = current
            # Strip T=1 dim that InferenceDataLoader adds; inner expects [B,N,C].
            batch_3d = {k: (v[:, 0] if isinstance(v, torch.Tensor) and v.ndim >= 3 and v.shape[1] == 1 else v)
                        for k, v in batch.items()}
            batch_3d[target] = current  # already 3D
            preds = inner(batch_3d)
            current = preds[target].detach()
            arr = current.cpu().numpy().reshape(args.n_inits, nlat, nlon, C)
            all_preds[:, 0, k] = arr
    wall = time.perf_counter() - t0
    logger.info("AR rollout done in %.1fs", wall)

    out_dir = EXP_DIR / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {target: (("init", "sample", "lead", "lat", "lon", "level"), all_preds),
         "obs_present": (("lead",), np.zeros(args.n_steps, dtype=bool))},
        coords={"init": np.array(init_indices), "sample": np.array([0]),
                "lead": np.arange(args.n_steps),
                "lat": loader.grid_info.lat, "lon": loader.grid_info.lon,
                "level": loader.grid_info.levels,
                "time": (("init", "lead"), time_coord)},
    )
    pf = out_dir / "preds_trajectory.zarr"
    if pf.exists(): import shutil; shutil.rmtree(pf)
    ds.to_zarr(pf, mode="w")

    # Reuse Phase 25b GT zarr (same init_indices, same target)
    src_gt = "/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25_transport_prior_osse/results/none_1month_v2/gt_trajectory.zarr"
    import shutil
    gt_path = out_dir / "gt_trajectory.zarr"
    if gt_path.exists(): shutil.rmtree(gt_path)
    shutil.copytree(src_gt, gt_path)
    gt_ds = xr.open_zarr(gt_path)

    pl, sm, rh = compute_trajectory_ensemble_metrics(gt_ds, ds, target_var=target)
    sd = out_dir / "scores"; sd.mkdir(exist_ok=True)
    pl.to_csv(sd / "metrics_per_lead.csv")
    sm.to_csv(sd / "metrics_summary.csv", header=["value"])
    pd.DataFrame(rh).to_csv(sd / "rank_histogram.csv", index_label="lead")

    logger.info("DET_BACKBONE RMSE=%.3f CRPS=%.3f spread/err=%.3f",
                float(sm["rmse_mean"]), float(sm["crps"]), float(sm["spread_error_ratio"]))
    logger.info("  RMSE@1=%.3f @28=%.3f @119=%.3f",
                float(pl["rmse_mean"].iloc[0]), float(pl["rmse_mean"].iloc[27]),
                float(pl["rmse_mean"].iloc[-1]))


if __name__ == "__main__":
    main()
