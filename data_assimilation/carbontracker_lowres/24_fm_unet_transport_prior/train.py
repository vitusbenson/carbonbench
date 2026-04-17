"""Phase 24: Conditional FM with transport priors.

Trains a flow matching UNet conditioned on the previous timestep's CO2 field
and wind fields (u, v). Reuses Phase 11's best hyperparameters from Optuna.

Channel layout fed to the UNet: [x_t (10), co2_t (10), u_t (10), v_t (10), time (1)] = 41.
Slot 0 is the "target-time placeholder" (co2massmix_next) that training_forward
overwrites with x_t. Slots 1-3 are genuine conditioning (preserved end-to-end).

Usage:
    python train.py --study-db ../11_fm_unet_final/optuna_study.db
    python train.py --study-db ../11_fm_unet_final/optuna_study.db --smoke
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
from neural_transport.training.tuning import MODEL_SIZES, get_best_config

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"

TARGET_VARS = ["co2massmix"]
FORCING_VARS = ["co2massmix", "u", "v"]  # prior CO2 + winds at time t
INPUT_VARS = ["co2massmix_next"] + FORCING_VARS  # _next slot is x_t placeholder

GRID = "latlon5.625"
VERTICAL_LEVELS = "l10"
FREQ = "6h"

EXP_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train Phase 24 FM w/ transport priors")
    parser.add_argument("--study-db", type=str, required=True,
                        help="Path to Phase 11 optuna_study.db to reuse best hparams")
    parser.add_argument("--study-name", type=str, default="fm_tuning")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-data-root", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="100 steps")
    parser.add_argument("--only-pred", action="store_true")
    parser.add_argument("--ckpt", type=str, default="best")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = 100

    data_root = args.training_data_root or DEFAULT_DATA_ROOT

    import optuna
    storage = f"sqlite:///{Path(args.study_db).resolve()}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    best = get_best_config(study)

    logger.info("Reusing Phase 11 best trial #%d (value=%.6f)",
                best["best_trial_number"], best["best_value"])

    nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[VERTICAL_LEVELS]["level"])
    lat = LATLON_PROTOTYPE_COORDS[GRID]["lat"]
    lon = LATLON_PROTOTYPE_COORDS[GRID]["lon"]

    N_INPUT_GROUPS = len(INPUT_VARS)           # 4 → co2_next placeholder + co2_t + u + v
    LEN_ALL_TARGET_VARS = nlev * len(TARGET_VARS)
    IN_CHANS = nlev * N_INPUT_GROUPS + 1        # 10*4 + 1 = 41
    OUT_CHANS = LEN_ALL_TARGET_VARS             # 10

    cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(len(lon), axis=1).reshape(-1, 1)
    cos_lat = cos_lat / np.mean(cos_lat)
    METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in TARGET_VARS}

    model_size_config = MODEL_SIZES[best["model_size"]]

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
        ),
    )

    loss_name = "flowmatching_weighted_mse" if best.get("time_loss_weight") else "flowmatching_mse"

    lit_module_kwargs = dict(
        model="flowmatching",
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
            halfcosine_steps=best.get("halfcosine_steps", 10000),
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
        log_every_n_steps=100,
        gradient_clip_val=best.get("gradient_clip_val", 32),
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
