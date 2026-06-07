"""P3.8 — Amortized conditional residual flow matching.

Fine-tunes the phase-2p residual-FM into an *amortized* posterior sampler: the
velocity UNet gains 2 input channels (obs_value, obs_mask) and is trained with a
randomly-masked nominal-column observation of the target, so it learns
``p(x_next | x_t, f_det, y)`` in a single generation pass (no test-time guidance).

Starts from a surgically channel-expanded checkpoint (make_init_ckpt.py) so the
dynamics are inherited and only the obs-conditioning is learned.

Usage:
    python make_init_ckpt.py            # once, builds init_amortized.ckpt
    python train.py --smoke             # 30-step sanity
    python train.py --max-steps 6000 --batch-size 96
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

DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker_leakfree"
TARGET_VARS = ["co2massmix"]
FORCING_VARS = ["co2massmix", "u", "v"]
INPUT_VARS = ["co2massmix_next"] + FORCING_VARS

GRID = "latlon5.625"
VERTICAL_LEVELS = "l10"
FREQ = "6h"
EXP_DIR = Path(__file__).resolve().parent
BASE = EXP_DIR.parent / "25c_v4_residual_fm" / "phase2p_residual_fm_stable_leakfree"

DEFAULT_DET_CKPT = str(
    EXP_DIR.parent / "25c_v4_residual_fm" / "phase1p_det_stable_rollout_ft_leakfree"
    / "singlestep" / "checkpoints" / "ema_frozen.ckpt"
)
DEFAULT_SIGMA_RES = str(BASE / "sigma_res.npy")
DEFAULT_INIT_CKPT = str(EXP_DIR / "init_amortized.ckpt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--training-data-root", type=str, default=None)
    p.add_argument("--det-ckpt", type=str, default=DEFAULT_DET_CKPT)
    p.add_argument("--sigma-res", type=str, default=DEFAULT_SIGMA_RES)
    p.add_argument("--init-ckpt", type=str, default=DEFAULT_INIT_CKPT)
    p.add_argument("--cov-lo", type=float, default=0.0)
    p.add_argument("--cov-hi", type=float, default=0.30)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.max_steps = 30
        args.batch_size = 8

    data_root = args.training_data_root or DEFAULT_DATA_ROOT

    import optuna
    storage = f"sqlite:///{EXP_DIR.parent / '11_fm_unet_final' / 'optuna_study.db'}"
    study = optuna.load_study(study_name="fm_tuning", storage=storage)
    best = get_best_config(study)

    nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[VERTICAL_LEVELS]["level"])
    lat = LATLON_PROTOTYPE_COORDS[GRID]["lat"]
    lon = LATLON_PROTOTYPE_COORDS[GRID]["lon"]

    cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(len(lon), axis=1).reshape(-1, 1)
    cos_lat = cos_lat / np.mean(cos_lat)
    METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in TARGET_VARS}

    model_size_config = MODEL_SIZES[best["model_size"]]

    N_INPUT_GROUPS = len(INPUT_VARS)
    LEN_ALL_TARGET_VARS = nlev * len(TARGET_VARS)
    # +2 obs channels (obs_value, obs_mask) vs the unconditional residual-FM.
    IN_CHANS = nlev * N_INPUT_GROUPS + nlev + 2 + 1   # 40 + 10 + 2 + 1 = 53
    OUT_CHANS = LEN_ALL_TARGET_VARS

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
            det_ckpt=args.det_ckpt,
            sigma_res_path=args.sigma_res,
            obs_coverage_range=(args.cov_lo, args.cov_hi),
        ),
    )

    loss_name = "flowmatching_weighted_mse" if best.get("time_loss_weight") else "flowmatching_mse"

    lit_module_kwargs = dict(
        model="amortized_residual_flowmatching",
        model_kwargs=wrapper_kwargs,
        loss=loss_name,
        loss_kwargs=dict(),
        metrics=[dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS)) for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]],
        no_grad_step_shedule=None,
        lr=best["lr"] * 0.3,  # gentler for fine-tuning
        weight_decay=best["weight_decay"],
        lr_shedule_kwargs=dict(
            warmup_steps=min(best["warmup_steps"], args.max_steps // 10),
            halfcosine_steps=args.max_steps,
            min_lr=1e-7,
            max_lr=best["max_lr"] * 0.3,
        ),
        val_dataloader_names=["singlestep"],
        plot_kwargs=dict(
            variables=["co2molemix"], layer_idxs=[0, 3, 5, 8], n_samples=2,
            dataset="carbontracker", grid=GRID, vertical_levels=VERTICAL_LEVELS, max_workers=32,
        ),
    )

    data_kwargs = dict(
        data_path=data_root, dataset="carbontracker", grid=GRID, vertical_levels=VERTICAL_LEVELS,
        freq=FREQ, n_timesteps=1, batch_size_train=args.batch_size, batch_size_pred=32, num_workers=0,
        val_rollout_n_timesteps=None, target_vars=TARGET_VARS, forcing_vars=FORCING_VARS, compute=True,
    )

    trainer_kwargs = dict(
        max_steps=args.max_steps, accelerator="gpu", devices=1, log_every_n_steps=50,
        gradient_clip_val=best.get("gradient_clip_val", 32), precision="bf16-mixed", strategy="auto",
    )

    init_ckpt = args.init_ckpt if (args.init_ckpt and Path(args.init_ckpt).exists()) else None
    if init_ckpt:
        logger.info("Fine-tuning from surgically-expanded init: %s", init_ckpt)
    else:
        logger.warning("No init ckpt found — training amortized head from scratch.")

    train_singlestep(
        EXP_DIR, data_kwargs, lit_module_kwargs, trainer_kwargs,
        ckptpath=init_ckpt,
        ckpt_kwargs=dict(
            save_top_k=3, save_last=True, monitor="Loss/Val_singlestep",
            filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_singlestep:.6f}",
            auto_insert_metric_name=False, every_n_epochs=1,
        ),
    )


if __name__ == "__main__":
    main()
