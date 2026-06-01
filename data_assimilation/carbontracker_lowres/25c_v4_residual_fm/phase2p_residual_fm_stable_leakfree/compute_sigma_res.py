"""Compute per-level standard deviation of (x_next - f_det(x)) over the training split.

Used by ResidualFlowMatching to standardize the FM target. Run once before
phase 2 training; saves a numpy array of shape ``(nlev,)`` to disk.

Usage:
    python compute_sigma_res.py \
        --det-ckpt ../phase1b_det_rollout_ft/singlestep/checkpoints/Epoch=13-Step=2716-LossVal=0.920315.ckpt \
        --output sigma_res.npy
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_env_bin = str(Path(sys.executable).parent)
if _env_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _env_bin + ":" + os.environ.get("PATH", "")

import numpy as np
import pytorch_lightning as pl
import torch

from neural_transport.datamodule import CarbonDataModule
from neural_transport.litmodule import NeuralTransport

DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
TARGET_VARS = ["co2massmix"]
FORCING_VARS = ["co2massmix", "u", "v"]
GRID = "latlon5.625"
VERTICAL_LEVELS = "l10"
FREQ = "6h"

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det-ckpt", type=str, required=True)
    p.add_argument("--output", type=str, default="sigma_res.npy")
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--max-batches", type=int, default=200,
                   help="Cap batches scanned (training split is large; ~200 batches @256 ~= 50k samples).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    # Load f_det.
    det_lit = NeuralTransport.load_from_checkpoint(
        args.det_ckpt, map_location="cpu", weights_only=False
    )
    f_det = det_lit.model.to(args.device).eval()
    target_var = TARGET_VARS[0]

    # Build a data module with single-step training pairs.
    dm = CarbonDataModule(
        data_path=args.data_root,
        dataset="carbontracker",
        grid=GRID,
        vertical_levels=VERTICAL_LEVELS,
        freq=FREQ,
        n_timesteps=1,
        batch_size_train=args.batch_size,
        batch_size_pred=args.batch_size,
        num_workers=0,
        target_vars=TARGET_VARS,
        forcing_vars=FORCING_VARS,
        compute=True,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    logger.info("Train dataloader: %d batches", len(loader))

    # Welford-style running variance per channel (sum, sum-sq, n).
    sum_sq = None
    sum_ = None
    n = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            # Move to device.
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device, non_blocking=True)
            # The litmodule.forward usually slices [:, t]; here we set
            # n_timesteps=1 so batch[var] has shape [B, 1, N, C]. Squeeze T.
            batch_t0 = {
                k: (v[:, 0] if (isinstance(v, torch.Tensor) and v.ndim >= 3 and v.shape[1] == 1) else v)
                for k, v in batch.items()
            }
            # Mimic AR-inference: replace `_next` placeholder with current state
            # so f_det's targshift uses the same mean it would at rollout time.
            batch_for_det = dict(batch_t0)
            if f"{target_var}_next" in batch_for_det:
                gt_next = batch_for_det[f"{target_var}_next"].clone()
                batch_for_det[f"{target_var}_next"] = batch_for_det[target_var].clone()
            else:
                raise RuntimeError(f"Missing {target_var}_next in batch")

            det_preds = f_det(batch_for_det)
            det_pred_phys = det_preds[target_var]  # [B, N, C]
            residual = (gt_next - det_pred_phys).float()

            # Reduce over (B, N) to per-channel statistics.
            B, N, C = residual.shape
            flat = residual.reshape(-1, C)
            if sum_ is None:
                sum_ = flat.sum(dim=0).cpu()
                sum_sq = (flat ** 2).sum(dim=0).cpu()
            else:
                sum_ += flat.sum(dim=0).cpu()
                sum_sq += (flat ** 2).sum(dim=0).cpu()
            n += flat.shape[0]
            if i % 25 == 0:
                logger.info("  batch %d/%d  n=%d", i, min(len(loader), args.max_batches), n)

    mean = (sum_ / n).numpy()
    var = (sum_sq / n).numpy() - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-20)).astype("float32")

    out_path = Path(args.output).resolve()
    np.save(out_path, std)
    logger.info("sigma_res per level (n=%d samples * gridcells): %s", n, np.array2string(std, precision=4))
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
