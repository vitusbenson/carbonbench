"""Baseline Flow Matching model training.

Trains a UNet-S FM model with safe defaults (GroupNorm, no OT coupling,
uniform time sampling) to establish a baseline before Optuna tuning.

Usage:
    python train.py                                          # Full (10k steps)
    python train.py --smoke                                  # Smoke test (100 steps)
    python train.py --max-steps 3000 --batch-size 256        # Quick run on small GPU
    python train.py --only-pred --ckpt best                  # Eval only
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from neural_transport.datasets.grids import (
    LATLON_PROTOTYPE_COORDS,
    VERTICAL_LAYERS_PROTOTYPE_COORDS,
)
from neural_transport.training import train_and_eval_singlestep
from neural_transport.training.ema import EMACallback
from neural_transport.training.tuning import MODEL_SIZES

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"

TARGET_VARS = ["co2massmix"]
GRID = "latlon5.625"
VERTICAL_LEVELS = "l10"
FREQ = "6h"

EXP_DIR = Path(__file__).resolve().parent

# ── Baseline config ──────────────────────────────────────────────────────

MODEL_SIZE = "S"
NORM = "group"
USE_OT_COUPLING = False
TIME_SAMPLING = "uniform"
TIME_LOSS_WEIGHT = None
LR = 3e-3
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 500
HALFCOSINE_STEPS = 10000
MAX_LR = 0.8
GRADIENT_CLIP_VAL = 32


def main():
    parser = argparse.ArgumentParser(description="Train baseline FM model")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--training-data-root", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="100 steps")
    parser.add_argument("--only-pred", action="store_true")
    parser.add_argument("--ckpt", type=str, default="best")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = 100

    data_root = args.training_data_root or DEFAULT_DATA_ROOT

    nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[VERTICAL_LEVELS]["level"])
    lat = LATLON_PROTOTYPE_COORDS[GRID]["lat"]
    lon = LATLON_PROTOTYPE_COORDS[GRID]["lon"]

    LEN_ALL_TARGET_VARS = nlev * len(TARGET_VARS)

    cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(len(lon), axis=1).reshape(-1, 1)
    cos_lat = cos_lat / np.mean(cos_lat)
    METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in TARGET_VARS}

    model_size_config = MODEL_SIZES[MODEL_SIZE]

    logger.info(
        "Baseline: model=%s (%s), norm=%s, ot=%s, time=%s, lr=%.4f, "
        "batch=%d, steps=%d",
        MODEL_SIZE, model_size_config, NORM, USE_OT_COUPLING,
        TIME_SAMPLING, LR, args.batch_size, args.max_steps,
    )

    regulargrid_kwargs = dict(
        input_vars=TARGET_VARS,
        target_vars=TARGET_VARS,
        nlat=len(lat),
        nlon=len(lon),
        predict_delta=False,
        add_surfflux=False,
        dt=60 * 60 * 6,
        massfixer="",
        targshift=True,
    )

    wrapper_kwargs = dict(
        **regulargrid_kwargs,
        model_kwargs=dict(
            submodel="unet",
            model_kwargs=dict(
                **regulargrid_kwargs,
                model_kwargs=dict(
                    in_chans=LEN_ALL_TARGET_VARS + 1,
                    out_chans=LEN_ALL_TARGET_VARS,
                    embed_dim=model_size_config["embed_dim"],
                    act="leakyrelu",
                    norm=NORM,
                    enc_filters=model_size_config["enc_filters"],
                    dec_filters=model_size_config["dec_filters"],
                    in_interpolation="bilinear",
                    out_interpolation="nearest-exact",
                    out_clip=None,
                ),
            ),
            generating=True,
            return_intermediates=True,
            method="midpoint",
            nlev=nlev,
            step_size=0.2,
            use_ot_coupling=USE_OT_COUPLING,
            time_sampling=TIME_SAMPLING,
            time_loss_weight=TIME_LOSS_WEIGHT,
        ),
    )

    lit_module_kwargs = dict(
        model="flowmatching",
        model_kwargs=wrapper_kwargs,
        loss="flowmatching_mse",
        loss_kwargs=dict(),
        metrics=[
            dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS))
            for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]
        ],
        no_grad_step_shedule=None,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        lr_shedule_kwargs=dict(
            warmup_steps=WARMUP_STEPS,
            halfcosine_steps=HALFCOSINE_STEPS,
            min_lr=1e-7,
            max_lr=MAX_LR,
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
        n_timesteps=1,
        batch_size_train=args.batch_size,
        batch_size_pred=32,
        num_workers=0,
        val_rollout_n_timesteps=None,
        target_vars=["co2massmix"],
        forcing_vars=[],
        compute=True,
    )

    trainer_kwargs = dict(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=100,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        precision="bf16-mixed",
        strategy="auto",
    )

    generate_kwargs = dict(
        n_samples=100,
        refine_start=0.9,
        avg_over_levels=False,
    )

    ema_callback = EMACallback(decay=0.9999, ema_start_step=1000)

    train_and_eval_singlestep(
        EXP_DIR,
        data_kwargs,
        lit_module_kwargs,
        trainer_kwargs,
        Path(data_root) / "test",
        device="cuda",
        freq="QS",
        train=not args.only_pred,
        ckpt=args.ckpt,
        ckpt_kwargs=dict(
            save_top_k=3,
            save_last=True,
            monitor="Loss/Val_singlestep",
            filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_singlestep:.6f}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
        ),
        generate_kwargs=generate_kwargs,
        distributional_eval=True,
        distributional_eval_kwargs=dict(n_gt_samples=50, n_gen_samples=200),
        extra_callbacks=[ema_callback],
    )


if __name__ == "__main__":
    main()
