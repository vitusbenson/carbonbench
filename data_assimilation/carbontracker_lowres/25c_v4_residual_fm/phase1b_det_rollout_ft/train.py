"""Phase 25c v4 — Phase 1: Deterministic backbone training with rollout fine-tune.

This is the proper ArchesWeatherGen decomposition:

  x_{t+1} = f_det(x_t)              # Phase 1 (this script)  — deterministic UNet
  r_{t+1} = (x_{t+1} - f_det(x_t)) / sigma
  r_hat   = g_FM(noise | x_t, f_det(x_t))   # Phase 2 — residual FM

Phase 1 trains a deterministic UNet predicting next-state CO2, with multi-step
rollout fine-tune (Brandstetter / AWG f_θ Phase 3 recipe):
  * single-step pretraining on Phase 24's data setup
  * rollout fine-tune on K ∈ {2,3,4} consecutive steps with quadratic discount
    `1/(1+i)²`, full-grad through the rollout

Usage:
    python train.py --smoke                  # 30-step sanity
    python train.py --max-steps 30000        # full single-step pretraining
    python train.py --rollout-only --max-steps 8000   # rollout FT phase
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
from neural_transport.training.train import train_singlestep
from neural_transport.training.tuning import MODEL_SIZES, get_best_config

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
TARGET_VARS = ["co2massmix"]
FORCING_VARS = ["co2massmix", "u", "v"]
INPUT_VARS = FORCING_VARS  # No FM placeholder slot; det model takes x_t + winds directly.
GRID = "latlon5.625"
VERTICAL_LEVELS = "l10"
FREQ = "6h"
EXP_DIR = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--rollout-only", action="store_true",
                   help="Skip single-step pretraining; run only rollout FT from given ckpt.")
    p.add_argument("--rollout-K", type=int, default=4,
                   help="Window length for rollout FT. K=4 = +1 day window @ 6h.")
    p.add_argument("--rollout-prob", type=float, default=0.5)
    p.add_argument("--rollout-curriculum-end", type=int, default=2000)
    p.add_argument("--lr-mult", type=float, default=1.0)
    p.add_argument("--training-data-root", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ckpt-from", type=str, default=None,
                   help="Warm-start from existing det ckpt (e.g. for rollout FT phase).")
    args = p.parse_args()

    if args.smoke:
        args.max_steps = 30
        args.batch_size = 8

    data_root = args.training_data_root or DEFAULT_DATA_ROOT

    # Reuse Phase 11 hyperparameters for fair comparison with FM.
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

    IN_CHANS = nlev * len(INPUT_VARS)  # no time slot, no placeholder
    OUT_CHANS = nlev * len(TARGET_VARS)

    # NOTE: This config uses MODELS["unet"] directly (no FlowMatching wrapper).
    # The lit-module loss will be standard MSE on co2massmix_next.
    model_kwargs = dict(
        input_vars=INPUT_VARS,
        target_vars=TARGET_VARS,
        nlat=len(lat),
        nlon=len(lon),
        nlev=nlev,
        predict_delta=False,   # full state; matches Phase 24 FM. Avoids gph_bottom dependency.
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

    # Rollout-FT: use no_grad_step_shedule to do K-step backprop with quadratic
    # discount via the existing litmodule.no_grad_shedule mechanism.
    no_grad_step_shedule = None
    if args.rollout_only:
        no_grad_step_shedule = dict(
            from_step=0,
            t_no_grad=[],  # all steps grad-enabled
        )

    n_timesteps = args.rollout_K if args.rollout_only else 1

    lit_module_kwargs = dict(
        model="unet",
        model_kwargs=model_kwargs,
        loss="mse",
        loss_kwargs=dict(
            weights={"co2massmix": cos_lat},
            normalize_batch=False,
        ),
        metrics=[
            dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS))
            for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]
        ],
        no_grad_step_shedule=no_grad_step_shedule,
        lr=best["lr"] * args.lr_mult,
        weight_decay=best["weight_decay"],
        lr_shedule_kwargs=dict(
            warmup_steps=max(50, best["warmup_steps"]),
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

    # Direct call to train_singlestep — skips post-train predict (which fails on
    # gph_bottom for the deterministic pipeline). We only need the trained ckpt.
    train_singlestep(
        EXP_DIR,
        data_kwargs,
        lit_module_kwargs,
        trainer_kwargs,
        ckptpath=args.ckpt_from,
        ckpt_kwargs=dict(
            save_top_k=2,
            save_last=True,
            monitor="Loss/Val_singlestep",
            filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_singlestep:.6f}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
        ),
    )


if __name__ == "__main__":
    main()
