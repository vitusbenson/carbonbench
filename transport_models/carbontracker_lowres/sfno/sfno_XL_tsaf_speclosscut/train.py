#!usr/bin/python

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from pytorch_lightning.profilers import AdvancedProfiler

from neural_transport.datasets.grids import (
    LATLON_PROTOTYPE_COORDS,
    VERTICAL_LAYERS_PROTOTYPE_COORDS,
)
from neural_transport.datasets.vars import *
from neural_transport.training import train_and_eval_rollout, train_and_eval_singlestep

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)

TARGET_VARS = ["co2massmix"]
FORCING_VARS_2D = [
    "blh",
    "cell_area",
    "co2flux_anthro",
    "co2flux_land",
    "co2flux_ocean",
    "orography",
    "tisr",
]
FORCING_VARS_3D = [
    "airmass",
    "gph_bottom",
    "gph_top",
    "p_bottom",
    "p_top",
    "q",
    "t",
    "u",
    "v",
]
grid = "latlon5.625"
vertical_levels = "l10"
freq = "6h"

nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[vertical_levels]["level"])

FORCING_VARS = FORCING_VARS_2D + FORCING_VARS_3D
LEN_ALL_TARGET_VARS = nlev * len(TARGET_VARS)
LEN_ALL_FORCING_VARS = len(FORCING_VARS_2D) + nlev * len(FORCING_VARS_3D)
LEN_ALL_VARS = LEN_ALL_TARGET_VARS + LEN_ALL_FORCING_VARS

lat = LATLON_PROTOTYPE_COORDS[grid]["lat"]
lon = LATLON_PROTOTYPE_COORDS[grid]["lon"]

cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(len(lon), axis=1).reshape(-1, 1)
cos_lat = cos_lat / np.mean(cos_lat)

ds_stats = xr.open_zarr(
    f"/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/train/carbontracker_{grid}_{vertical_levels}_{freq}_stats.zarr"
).compute()

inv_std = {
    k: 1
    / (ds_stats[f"{k}_delta"].sel(stats="std").where(lambda x: x > 1e-14, 1).values)
    ** 2
    for k in TARGET_VARS  # CARBOSCOPE_CARBON3D_VARS
}

weights = {k: cos_lat * inv_std[k] for k in inv_std}

LOSS_WEIGHTS = {k: (10 * v / LEN_ALL_TARGET_VARS) for k, v in weights.items()}

METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in TARGET_VARS}


MODEL_DIMS = {
    "XS": dict(embed_dim=64, num_layers=4),
    "S": dict(embed_dim=128, num_layers=4),
    "M": dict(embed_dim=128, num_layers=8),
    "L": dict(embed_dim=256, num_layers=8),
    "XL": dict(embed_dim=256, num_layers=12),
}

MODEL_SIZE = "XL"

model_kwargs = dict(
    model_kwargs=dict(
        embed_dim=MODEL_DIMS[MODEL_SIZE]["embed_dim"],
        num_layers=MODEL_DIMS[MODEL_SIZE]["num_layers"],
        scale_factor=1,
        in_chans=LEN_ALL_VARS,
        out_chans=LEN_ALL_TARGET_VARS,
        normalization_layer="instance_norm",
        rank=1,
    ),
    input_vars=TARGET_VARS + FORCING_VARS,
    target_vars=TARGET_VARS,
    nlat=len(lat),
    nlon=len(lon),
    predict_delta=True,
    add_surfflux=True,
    dt=60 * 60 * 6,
    massfixer="scale",
    targshift=True,
)

lit_module_kwargs = dict(
    model="sfno",
    model_kwargs=model_kwargs,
    loss="mse",
    loss_kwargs=dict(
        weights=LOSS_WEIGHTS,
        spectral_power_weight=0.1,
        nlat=len(lat),
        nlon=len(lon),
        cutoff=(len(lat) // 4) * 3,
    ),
    metrics=[
        dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS))
        for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]
    ]
    + [dict(name="mass_rmsev2", kwargs=dict(molecule=m)) for m in ["co2"]],
    no_grad_step_shedule=None,
    lr=1e-3,
    weight_decay=0.1,
    lr_shedule_kwargs=dict(
        warmup_steps=1000, halfcosine_steps=99000, min_lr=3e-7, max_lr=1.0
    ),
    val_dataloader_names=["singlestep", "rollout"],  #
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
    data_path="/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker",
    dataset="carbontracker",
    grid=grid,
    vertical_levels=vertical_levels,
    freq=freq,
    n_timesteps=1,
    batch_size_train=BATCH_SIZE_TRAIN // N_GPUS,
    batch_size_pred=BATCH_SIZE_PRED,
    num_workers=32 * N_GPUS,
    val_rollout_n_timesteps=31,
    target_vars=["co2massmix", "airmass"],
    forcing_vars=[
        "gph_bottom",
        "gph_top",
        "p_bottom",
        "p_top",
        "q",
        "t",
        "u",
        "v",
        "blh",
        "cell_area",
        "co2flux_anthro",
        "co2flux_land",
        "co2flux_ocean",
        "orography",
        "tisr",
    ],
    compute=False,
    # time_interval=["1990-01-01", "2014-12-31"],
)

data_path_forecast = Path(
    "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/"
)


trainer_kwargs = dict(
    max_steps=100000,
    accelerator="gpu",
    devices=N_GPUS,
    log_every_n_steps=100,
    gradient_clip_val=32,
    # precision="bf16-mixed",
    strategy=(
        "auto"
        if N_GPUS == 1
        else pl.strategies.DDPStrategy(find_unused_parameters=False)
    ),
    # profiler="simple"
    # fast_dev_run=True,
)

rollout_trainer_kwargs = dict(
    max_epochs=2,
    accelerator="gpu",
    devices=N_GPUS,
    log_every_n_steps=50,
    gradient_clip_val=32,
    # precision="bf16-mixed",
    strategy=(
        "auto"
        if N_GPUS == 1
        else pl.strategies.DDPStrategy(find_unused_parameters=False)
    ),
)

obs_compare_path = f"/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/obs_carbontracker_{grid}_{vertical_levels}_{freq}.zarr"


def main(rollout=False, train=True, ckpt="last", data_path=None):
    run_dir = Path(__file__).resolve().parent

    if data_path is not None:
        data_kwargs["data_path"] = data_path

    if rollout:
        train_and_eval_rollout(
            run_dir,
            data_kwargs,
            lit_module_kwargs,
            rollout_trainer_kwargs,
            data_path_forecast,
            device="cuda",
            freq="QS",
            obs_compare_path=obs_compare_path,
            movie_interval=["2018-01-01", "2018-03-31"],
            num_workers=32,
            rollout_constant_lr=1e-5,
            timesteps=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            train=train,
            ckpt=ckpt,
            massfixers=[None, "scale"],
        )

    else:
        train_and_eval_singlestep(
            run_dir,
            data_kwargs,
            lit_module_kwargs,
            trainer_kwargs,
            data_path_forecast,
            device="cuda",
            freq="QS",
            obs_compare_path=obs_compare_path,
            movie_interval=["2018-01-01", "2018-03-31"],
            num_workers=32,
            train=train,
            ckpt=ckpt,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--only_pred", action="store_true")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    main(
        rollout=args.rollout,
        train=not args.only_pred,
        ckpt=args.ckpt,
        data_path=args.data_path,
    )
