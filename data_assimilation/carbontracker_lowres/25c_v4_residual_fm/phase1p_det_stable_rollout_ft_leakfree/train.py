"""Phase 25p — STABLE rollout fine-tune of the deterministic backbone.

Why a new phase: phase1b val-loss went UP during 8k rollout-FT (0.92 → 0.94)
because the recipe re-used the single-step max_lr (0.73) — way too aggressive
for fine-tune from a converged ckpt. Quadratic discount was a TODO and never
applied; K=4 ran without curriculum.

Fixes (this script):
  * lr_mult default 0.2 (max_lr ≈ 0.15 instead of 0.73).
  * Quadratic step-discount weights `1/(1+i)^2` — implemented in MSE loss.
    Step 0 (single-step) gets 2.82× the weight of step 3, so single-step
    quality stays anchored.
  * EMA on f_det (decay 0.999, start step 200 — early enough to capture the
    fine-tune trajectory, late enough to skip the warmup transient).
  * K=4 fixed (the discount makes a curriculum unnecessary — step 3 is
    already weighted at 0.18).
  * Warm-start ckpt MUST be supplied (--ckpt-from /.../phase1_det_backbone/.../best.ckpt).

Usage:
    python train.py --smoke --ckpt-from /path/to/phase1_best.ckpt
    python train.py --max-steps 8000 --ckpt-from /path/to/phase1_best.ckpt
"""

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

from neural_transport.datasets.grids import (
    LATLON_PROTOTYPE_COORDS,
    VERTICAL_LAYERS_PROTOTYPE_COORDS,
)
from neural_transport.training.ema import EMACallback
from neural_transport.training.train import train_singlestep
from neural_transport.training.tuning import MODEL_SIZES, get_best_config

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
TARGET_VARS = ["co2massmix"]
FORCING_VARS = ["co2massmix", "u", "v"]
INPUT_VARS = FORCING_VARS
GRID = "latlon5.625"
VERTICAL_LEVELS = "l10"
FREQ = "6h"
EXP_DIR = Path(__file__).resolve().parent

DEFAULT_CKPT_FROM = (
    "/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/"
    "data_assimilation/carbontracker_lowres/25c_v4_residual_fm/"
    "phase1_det_backbone/singlestep/checkpoints/best.ckpt"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--rollout-K", type=int, default=4)
    p.add_argument("--lr-mult", type=float, default=0.2)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-start", type=int, default=200)
    p.add_argument("--no-step-discount", action="store_true",
                   help="Disable the 1/(1+i)^2 quadratic discount (uniform weights).")
    p.add_argument("--training-data-root", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ckpt-from", type=str, default=DEFAULT_CKPT_FROM,
                   help="Warm-start ckpt (single-step pretrained det backbone).")
    args = p.parse_args()

    if args.smoke:
        args.max_steps = 30
        args.batch_size = 8

    data_root = args.training_data_root or DEFAULT_DATA_ROOT

    import optuna
    storage = f"sqlite:///{EXP_DIR.parent.parent / '11_fm_unet_final' / 'optuna_study.db'}"
    study = optuna.load_study(study_name="fm_tuning", storage=storage)
    best = get_best_config(study)

    nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[VERTICAL_LEVELS]["level"])
    lat = LATLON_PROTOTYPE_COORDS[GRID]["lat"]
    lon = LATLON_PROTOTYPE_COORDS[GRID]["lon"]

    cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(len(lon), axis=1).reshape(-1, 1)
    cos_lat = cos_lat / np.mean(cos_lat)
    METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in TARGET_VARS}

    model_size_config = MODEL_SIZES[best["model_size"]]

    IN_CHANS = nlev * len(INPUT_VARS)
    OUT_CHANS = nlev * len(TARGET_VARS)

    model_kwargs = dict(
        input_vars=INPUT_VARS,
        target_vars=TARGET_VARS,
        nlat=len(lat),
        nlon=len(lon),
        nlev=nlev,
        predict_delta=False,
        add_surfflux=False,
        dt=60 * 60 * 6,
        massfixer="",
        targshift=True,
        model_kwargs=dict(
            in_chans=IN_CHANS,
            out_chans=OUT_CHANS,
            embed_dim=model_size_config["embed_dim"],
            act="leakyrelu",
            norm=best.get("norm", "group"),
            enc_filters=model_size_config["enc_filters"],
            dec_filters=model_size_config["dec_filters"],
            in_interpolation="bilinear",
            out_interpolation="nearest-exact",
            out_clip=None,
        ),
    )

    no_grad_step_shedule = dict(from_step=0, t_no_grad=[])  # all rollout steps grad-enabled
    n_timesteps = args.rollout_K

    loss_kwargs = dict(
        weights={"co2massmix": cos_lat},
        normalize_batch=False,
    )
    if not args.no_step_discount:
        loss_kwargs["step_weights"] = "quadratic_discount"

    lit_module_kwargs = dict(
        model="unet",
        model_kwargs=model_kwargs,
        loss="mse",
        loss_kwargs=loss_kwargs,
        metrics=[
            dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS))
            for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]
        ],
        no_grad_step_shedule=no_grad_step_shedule,
        lr=best["lr"] * args.lr_mult,
        weight_decay=best["weight_decay"],
        lr_shedule_kwargs=dict(
            warmup_steps=max(50, int(best["warmup_steps"] * 0.5)),
            halfcosine_steps=args.max_steps,
            min_lr=1e-7,
            max_lr=best["max_lr"] * args.lr_mult,
        ),
        val_dataloader_names=["singlestep"],
        plot_kwargs=dict(
            variables=["co2molemix"],
            layer_idxs=[0, 3, 5, 8],
            n_samples=2,
            dataset="carbontracker",
            grid=GRID,
            vertical_levels=VERTICAL_LEVELS,
            max_workers=32,
        ),
    )

    data_kwargs = dict(
        data_path=data_root,
        dataset="carbontracker",
        grid=GRID,
        vertical_levels=VERTICAL_LEVELS,
        freq=FREQ,
        n_timesteps=n_timesteps,
        batch_size_train=args.batch_size,
        batch_size_pred=32,
        num_workers=0,
        val_rollout_n_timesteps=None,
        target_vars=TARGET_VARS,
        forcing_vars=FORCING_VARS,
        compute=True,
    )

    trainer_kwargs = dict(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=50,
        gradient_clip_val=best.get("gradient_clip_val", 32),
        precision="bf16-mixed",
        strategy="auto",
    )

    ema_cb = EMACallback(decay=args.ema_decay, ema_start_step=args.ema_start)

    train_singlestep(
        EXP_DIR,
        data_kwargs,
        lit_module_kwargs,
        trainer_kwargs,
        ckptpath=args.ckpt_from,
        ckpt_kwargs=dict(
            save_top_k=3,
            save_last=True,
            monitor="Loss/Val_singlestep",
            filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_singlestep:.6f}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
        ),
        extra_callbacks=[ema_cb],
    )


if __name__ == "__main__":
    main()
