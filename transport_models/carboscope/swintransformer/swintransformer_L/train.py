#!usr/bin/python

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from pytorch_lightning.profilers import AdvancedProfiler

from neural_transport.datasets.vars import (
    CARBOSCOPE_CARBON2D_VARS,
    CARBOSCOPE_CARBON3D_VARS,
    CARBOSCOPE_METEO2D_VARS,
    CARBOSCOPE_METEO3D_VARS,
)
from neural_transport.training import train_and_eval_rollout, train_and_eval_singlestep

torch.set_float32_matmul_precision("high")

TARGET_VARS = ["co2massmix"]
FORCING_VARS = ["cell_area"] + CARBOSCOPE_CARBON2D_VARS + CARBOSCOPE_METEO3D_VARS
LEN_ALL_TARGET_VARS = 19 * len(["co2massmix"])
LEN_ALL_FORCING_VARS = (
    19 * len(CARBOSCOPE_METEO3D_VARS) + len(CARBOSCOPE_CARBON2D_VARS) + 1
)
LEN_ALL_VARS = LEN_ALL_TARGET_VARS + LEN_ALL_FORCING_VARS


cos_lat = (
    np.cos(np.radians(np.linspace(-88, 88, 45)))[:, None, None]
    .repeat(72, axis=1)
    .reshape(-1, 1)
)
cos_lat = cos_lat / np.mean(cos_lat)

ds_stats = xr.open_dataset(
    "/User/homes/vbenson/vbenson/graph_tm/data/Carboscope/train/carboscope_stats.nc"
)

inv_std = {
    k: 1
    / (ds_stats[f"{k}_delta"].sel(stats="std").where(lambda x: x > 1e-14, 1).values)
    ** 2
    for k in TARGET_VARS  # CARBOSCOPE_CARBON3D_VARS
}

weights = {k: cos_lat * inv_std[k] for k in inv_std}

LOSS_WEIGHTS = {
    k: (
        v / LEN_ALL_TARGET_VARS
        if k in CARBOSCOPE_CARBON2D_VARS
        else 19 * v / LEN_ALL_TARGET_VARS
    )
    for k, v in weights.items()
}

METRIC_WEIGHTS = {f"{k}_delta": cos_lat for k in TARGET_VARS}


MODEL_DIMS = {
    "S": dict(embed_dim=512, depth=12, img_window_ratio=8),
    "M": dict(embed_dim=768, depth=12, img_window_ratio=8),
    "L": dict(embed_dim=1024, depth=24, img_window_ratio=4),
}

MODEL_SIZE = "L"

model_kwargs = dict(
    model_kwargs=dict(
        img_size=(64, 96),
        patch_size=4,
        embed_dim=MODEL_DIMS[MODEL_SIZE]["embed_dim"],
        depths=(MODEL_DIMS[MODEL_SIZE]["depth"],),
        in_chans=LEN_ALL_VARS,
        out_chans=LEN_ALL_TARGET_VARS,
        num_heads=(8,),
        img_window_ratio=MODEL_DIMS[MODEL_SIZE]["img_window_ratio"],
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        interpolation_mode="bilinear",
    ),
    input_vars=TARGET_VARS + FORCING_VARS,
    target_vars=TARGET_VARS,
    nlat=45,
    nlon=72,
    predict_delta=True,
    add_surfflux=True,
    dt=3600,
)

lit_module_kwargs = dict(
    model="swintransformer",
    model_kwargs=model_kwargs,
    loss="mse",
    loss_kwargs=dict(weights=LOSS_WEIGHTS),
    metrics=[
        dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS))
        for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]
    ]
    + [
        dict(name="mass_rmse", kwargs=dict(molecule=m, weights=cos_lat))
        for m in ["co2"]
    ],
    no_grad_step_shedule=None,
    lr=1e-3,
    weight_decay=0.1,
    lr_shedule_kwargs=dict(
        warmup_steps=1000, halfcosine_steps=299000, min_lr=3e-7, max_lr=1.0
    ),
    val_dataloader_names=["singlestep", "rollout"],  #
    plot_kwargs=dict(
        variables=["co2molemix"],
        layer_idxs=[0, 1, 9, 15],
        n_samples=4,
        dataset="carboscope",
        grid="latlon4",
        max_workers=32,
    ),
)

N_GPUS = 1
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_PRED = 32

data_kwargs = dict(
    data_path="/scratch/vbenson/graph_tm/data/Carboscope",
    dataset="carboscope",
    grid="latlon4",
    n_timesteps=1,
    batch_size_train=BATCH_SIZE_TRAIN // N_GPUS,
    batch_size_pred=BATCH_SIZE_PRED,
    num_workers=32 * N_GPUS,
    val_rollout_n_timesteps=31,
    target_vars=["co2density", "airdensity", "volume"],
    forcing_vars=[
        "gp",
        "omeg",
        "q",
        "r",
        "t",
        "u",
        "v",
        "co2flux_land",
        "co2flux_ocean",
        "co2flux_subt",
        "cell_area",
    ],
)

data_path_forecast = Path("/User/homes/vbenson/vbenson/graph_tm/data/Carboscope/test/")


trainer_kwargs = dict(
    max_steps=300000,
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
    # profiler="simple"
    # fast_dev_run=True,
)

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

obs_compare_path = (
    "/User/homes/vbenson/vbenson/graph_tm/data/Carboscope/test/obspack_carboscope.zarr"
)


def main(rollout=False):
    run_dir = Path(__file__).resolve().parent

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
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", action="store_true")
    args = parser.parse_args()
    main(rollout=args.rollout)
