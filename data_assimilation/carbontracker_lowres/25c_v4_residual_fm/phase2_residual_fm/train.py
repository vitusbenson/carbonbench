"""Phase 25c v4 — Phase 2: Residual flow matching head.

Trains a flow matching UNet to model the *residual* on top of a frozen
deterministic backbone (Phase 1b best.ckpt). Inference combines the two:

    x_next = f_det(x) + sigma_res * g_FM(noise | x_t, x_data, f_det(x))

Channel layout fed to the velocity UNet (51 channels at l10):
    [x_t (residual at FM time t, 10) ,
     co2_t (10), u_t (10), v_t (10),
     f_det_pred normalized + targshifted (10),
     time (1)]

Usage:
    python train.py --smoke                                # 30-step sanity
    python train.py --max-steps 20000 --batch-size 128     # full run
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
INPUT_VARS = ["co2massmix_next"] + FORCING_VARS  # _next slot is residual placeholder

GRID = "latlon5.625"
VERTICAL_LEVELS = "l10"
FREQ = "6h"
EXP_DIR = Path(__file__).resolve().parent

DEFAULT_DET_CKPT = (
    "/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/"
    "data_assimilation/carbontracker_lowres/25c_v4_residual_fm/"
    "phase1b_det_rollout_ft/singlestep/checkpoints/"
    "Epoch=13-Step=2716-LossVal=0.920315.ckpt"
)
DEFAULT_SIGMA_RES = str(EXP_DIR / "sigma_res.npy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--training-data-root", type=str, default=None)
    p.add_argument("--det-ckpt", type=str, default=DEFAULT_DET_CKPT)
    p.add_argument("--sigma-res", type=str, default=DEFAULT_SIGMA_RES)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.max_steps = 30
        args.batch_size = 8

    data_root = args.training_data_root or DEFAULT_DATA_ROOT

    # Reuse Phase 11 best Optuna hyperparameters.
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

    # Channel layout: standard FM (4 groups x nlev) + det_pred (1 group x nlev) + time(1).
    N_INPUT_GROUPS = len(INPUT_VARS)              # 4 → co2_next placeholder + co2_t + u + v
    LEN_ALL_TARGET_VARS = nlev * len(TARGET_VARS)
    IN_CHANS = nlev * N_INPUT_GROUPS + nlev + 1   # 10*4 + 10 + 1 = 51
    OUT_CHANS = LEN_ALL_TARGET_VARS               # 10

    regulargrid_kwargs = dict(
        input_vars=INPUT_VARS,
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
            ),
            generating=True,
            return_intermediates=True,
            method="midpoint",
            nlev=nlev,
            step_size=0.2,
            use_ot_coupling=best["use_ot_coupling"],
            time_sampling=best["time_sampling"],
            time_sampling_kwargs=best.get("time_sampling_kwargs"),
            time_loss_weight=best.get("time_loss_weight"),
            # Phase 25g residual head specifics:
            det_ckpt=args.det_ckpt,
            sigma_res_path=args.sigma_res,
        ),
    )

    loss_name = "flowmatching_weighted_mse" if best.get("time_loss_weight") else "flowmatching_mse"

    lit_module_kwargs = dict(
        model="residual_flowmatching",
        model_kwargs=wrapper_kwargs,
        loss=loss_name,
        loss_kwargs=dict(),
        metrics=[
            dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS))
            for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]
        ],
        no_grad_step_shedule=None,
        lr=best["lr"],
        weight_decay=best["weight_decay"],
        lr_shedule_kwargs=dict(
            warmup_steps=best["warmup_steps"],
            halfcosine_steps=args.max_steps,
            min_lr=1e-7,
            max_lr=best["max_lr"],
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

    train_singlestep(
        EXP_DIR,
        data_kwargs,
        lit_module_kwargs,
        trainer_kwargs,
        ckpt_kwargs=dict(
            save_top_k=3,
            save_last=True,
            monitor="Loss/Val_singlestep",
            filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_singlestep:.6f}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
        ),
    )


if __name__ == "__main__":
    main()
