"""
Flow Matching training with minibatch OT coupling.

Based on 01_fm_unet_training_baseline/train.py with OT coupling enabled.
All other hyperparameters identical for fair comparison.

Usage:
    python train.py                    # Full training (10k steps)
    python train.py --max_steps 100    # Smoke test (100 steps)
    python train.py --only_pred        # Evaluation only
"""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

from neural_transport.datasets.grids import (
    LATLON_PROTOTYPE_COORDS,
    VERTICAL_LAYERS_PROTOTYPE_COORDS,
)
from neural_transport.datasets.vars import *  # noqa: F403
from neural_transport.training import train_and_eval_rollout, train_and_eval_singlestep

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)

DEFAULT_TRAINING_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
DEFAULT_MASKING_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/OCO2MIP_OCO2"
DEFAULT_FORECAST_DATA_ROOT = DEFAULT_TRAINING_DATA_ROOT + "/test"

TARGET_VARS = ["co2massmix"]
FORCING_VARS_1D = []
FORCING_VARS_2D = []
FORCING_VARS_3D = []
grid = "latlon5.625"
vertical_levels = "l10"
freq = "6h"

nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[vertical_levels]["level"])

FORCING_VARS = FORCING_VARS_1D + FORCING_VARS_2D + FORCING_VARS_3D
LEN_ALL_TARGET_VARS = nlev * len(TARGET_VARS)
LEN_ALL_FORCING_VARS = len(FORCING_VARS_1D) + len(FORCING_VARS_2D) + nlev * len(FORCING_VARS_3D)
LEN_ALL_VARS = LEN_ALL_TARGET_VARS + LEN_ALL_FORCING_VARS

lat = LATLON_PROTOTYPE_COORDS[grid]["lat"]
lon = LATLON_PROTOTYPE_COORDS[grid]["lon"]

cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(len(lon), axis=1).reshape(-1, 1)
cos_lat = cos_lat / np.mean(cos_lat)

ds_stats = xr.open_zarr(
    f"{DEFAULT_TRAINING_DATA_ROOT}/train/carbontracker_{grid}_{vertical_levels}_{freq}_stats.zarr"
).compute()

inv_std = {
    k: 1
    / (ds_stats[f"{k}_delta"].sel(stats="std").where(lambda x: x > 1e-14, 1).values)
    ** 2
    for k in TARGET_VARS
}

weights = {k: cos_lat * inv_std[k] for k in inv_std}

LOSS_WEIGHTS = {k: (10 * v / LEN_ALL_TARGET_VARS) for k, v in weights.items()}

METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in TARGET_VARS}


MODEL_DIMS = {
    "XS": dict(embed_dim=64),
    "S": dict(embed_dim=128),
    "M": dict(embed_dim=256),
    "L": dict(embed_dim=512),
    "XL": dict(embed_dim=1024),
}

MODEL_SIZE = "S"

regulargrid_kwargs = dict(
    input_vars=TARGET_VARS + FORCING_VARS,
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
                in_chans=LEN_ALL_VARS + 1,  # + 1 for flow_time
                out_chans=LEN_ALL_TARGET_VARS,
                embed_dim=MODEL_DIMS[MODEL_SIZE]["embed_dim"],
                act="leakyrelu",
                norm="batch",
                enc_filters=[[7], [3, 3], [3, 3], [3, 3]],
                dec_filters=[[3, 3], [3, 3], [3, 3], [3, 3]],
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
        use_ot_coupling=True,  # <-- OT coupling enabled
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
    lr=3e-3,
    weight_decay=0.0,
    lr_shedule_kwargs=dict(
        warmup_steps=5000, halfcosine_steps=80000, min_lr=1.448612222179744e-07, max_lr=0.8001606601982787
    ),
    val_dataloader_names=["singlestep"],
    plot_kwargs=dict(
        variables=["co2molemix"],
        layer_idxs=[0, 3, 5, 8],
        n_samples=2,
        dataset="carbontracker",
        grid=grid,
        vertical_levels=vertical_levels,
        max_workers=32,
    ),
)

N_GPUS = 1
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_PRED = 32

data_kwargs = dict(
    data_path=DEFAULT_TRAINING_DATA_ROOT,
    dataset="carbontracker",
    grid=grid,
    vertical_levels=vertical_levels,
    freq=freq,
    n_timesteps=1,
    batch_size_train=BATCH_SIZE_TRAIN // N_GPUS,
    batch_size_pred=BATCH_SIZE_PRED,
    num_workers=32 * N_GPUS,
    val_rollout_n_timesteps=None,
    target_vars=["co2massmix"],
    forcing_vars=[],
    compute=False,
)

data_path_forecast = Path(DEFAULT_FORECAST_DATA_ROOT)

generate_kwargs_nested = dict(
    general=dict(
        n_samples=100,
        refine_start=0.9,
        avg_over_levels=False,
    ),
    mask=dict(
        masking=True,
        mask_source="test",
        mask_pattern="vertical",
        masking_time=None,
        t_threshold=0.9,
        masking_method="interpolate",
        analyze_masking=True,
        obs_fraction=0.3,
    ),
    noise=dict(
        noise_pattern=None,
        analyze_noise=False,
    ),
)

obs_compare_path = f"{DEFAULT_FORECAST_DATA_ROOT}/obs_carbontracker_{grid}_{vertical_levels}_{freq}.zarr"


def flatten_dict(d: dict) -> dict:
    out = {}
    for section in d.values():
        out.update(section)
    return out


def main(
        rollout: bool = False,
        train: bool = True,
        ckpt: str = "last",
        max_steps: int = 10000,
        training_data_root: str | None = None,
        masking_data_root: str | None = None,
        forecast_data_root: str | None = None,
        ) -> None:
    run_dir = Path(__file__).resolve().parent

    generate_kwargs = flatten_dict(generate_kwargs_nested)

    if training_data_root is not None:
        data_kwargs["data_path"] = training_data_root
    if masking_data_root is not None:
        generate_kwargs["data_path_generate"] = masking_data_root + "/train"
    if forecast_data_root is not None:
        data_path_forecast_local = Path(forecast_data_root)
        obs_compare_path_local = f"{forecast_data_root}/obs_carbontracker_{grid}_{vertical_levels}_{freq}.zarr"
    else:
        data_path_forecast_local = data_path_forecast
        obs_compare_path_local = obs_compare_path

    trainer_kwargs = dict(
        max_steps=max_steps,
        accelerator="gpu",
        devices=N_GPUS,
        log_every_n_steps=100,
        gradient_clip_val=32,
        precision="bf16-mixed",
        strategy=(
            "auto"
            if N_GPUS == 1
            else pl.strategies.DDPStrategy(find_unused_parameters=False)
        ),
    )

    if rollout:
        rollout_trainer_kwargs = dict(
            max_epochs=2,
            accelerator="gpu",
            devices=N_GPUS,
            log_every_n_steps=50,
            gradient_clip_val=32,
            precision="bf16-mixed",
            strategy=(
                "auto"
                if N_GPUS == 1
                else pl.strategies.DDPStrategy(find_unused_parameters=False)
            ),
        )
        train_and_eval_rollout(
            run_dir,
            data_kwargs,
            lit_module_kwargs,
            rollout_trainer_kwargs,
            data_path_forecast_local,
            device="cuda",
            freq="QS",
            obs_compare_path=obs_compare_path_local,
            movie_interval=["2018-01-01", "2018-03-31"],
            num_workers=32,
            rollout_constant_lr=1e-5,
            timesteps=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            train=train,
            ckpt=ckpt,
            massfixers=["scale"],
            generate_kwargs=generate_kwargs,
        )
    else:
        train_and_eval_singlestep(
            run_dir,
            data_kwargs,
            lit_module_kwargs,
            trainer_kwargs,
            data_path_forecast_local,
            device="cuda",
            freq="QS",
            obs_compare_path=obs_compare_path_local,
            movie_interval=["2018-01-01", "2018-03-31"],
            num_workers=32,
            train=train,
            ckpt=ckpt,
            ckpt_kwargs=dict(
                save_top_k=1,
                save_last=True,
                monitor="Loss/Val_singlestep",
                filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_singlestep:.6f}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
            ),
            generate_kwargs=generate_kwargs,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--only_pred", action="store_true")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--training_data_root", type=str, default=str(DEFAULT_TRAINING_DATA_ROOT))
    parser.add_argument("--masking_data_root", type=str, default=str(DEFAULT_MASKING_DATA_ROOT))
    parser.add_argument("--forecast_data_root", type=str, default=str(DEFAULT_FORECAST_DATA_ROOT))
    args = parser.parse_args()
    main(
        rollout=args.rollout,
        train=not args.only_pred,
        ckpt=args.ckpt,
        max_steps=args.max_steps,
        training_data_root=args.training_data_root,
        masking_data_root=args.masking_data_root,
        forecast_data_root=args.forecast_data_root,
    )
