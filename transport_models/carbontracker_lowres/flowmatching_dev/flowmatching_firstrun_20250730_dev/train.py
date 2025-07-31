"""
Training and evaluation script for Flow Matching models on given data.

This script sets up the model architecture, training configuration,
and dataset parameters for predicting atmospheric CO2 mass mixing ratios
using the Flow Matching method implemented in the `neural_transport` framework.

It supports both single-step training/evaluation and multi-step rollout evaluation.
Execution mode is selected via command-line arguments.

Typical usage:
    python script_name.py --rollout        # Run rollout evaluation
    python script_name.py --only_pred      # Only evaluate, no training
    python script_name.py --ckpt best      # Use 'best' checkpoint instead of default 'last'
    python script_name.py --data_path ...  # Override dataset path
"""

#!usr/bin/python

from pathlib import Path

import numpy as np
import pytorch_lightning as pl

# torch
import torch
import xarray as xr
from neural_transport.datasets.grids import (
    LATLON_PROTOTYPE_COORDS,
    VERTICAL_LAYERS_PROTOTYPE_COORDS,
)
from neural_transport.datasets.vars import *  # noqa: F403

# neural_transport
from neural_transport.models.wrappers_registry import MODELWRAPPERS
from neural_transport.training import train_and_eval_rollout, train_and_eval_singlestep

torch.set_float32_matmul_precision("high")
pl.seed_everything(42)

TARGET_VARS = ["co2massmix"]
# Uncomment for conditional Flow Matching
FORCING_VARS_1D = [
    # "flow_time"
]
FORCING_VARS_2D = [
    # "blh",
    # "cell_area",
    # "co2flux_anthro",
    # "co2flux_land",
    # "co2flux_ocean",
    # "orography",
    # "tisr",
]
FORCING_VARS_3D = [
    # "airmass",
    # "gph_bottom",
    # "gph_top",
    # "p_bottom",
    # "p_top",
    # "q",
    # "t",
    # "u",
    # "v",
]
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
    "XS": dict(embed_dim=64),
    "S": dict(embed_dim=128),
    "M": dict(embed_dim=256),
    "L": dict(embed_dim=512),
    "XL": dict(embed_dim=1024),
}

MODEL_SIZE = "S"

regulargrid_kwargs = dict( # for RegularGridModel
    input_vars=TARGET_VARS + FORCING_VARS,
    target_vars=TARGET_VARS,
    nlat=len(lat),
    nlon=len(lon),
    predict_delta=False,
    add_surfflux=False,
    dt=60 * 60 * 6,
    massfixer="",
    targshift=False,
)

wrapper_kwargs = dict( # for RegularGridModel (FlowMatching)
    **regulargrid_kwargs,
    model_kwargs=dict( # for FlowMatching
        submodel="unet",
        model_kwargs=dict( # for RegularGridModel (UNet)
            **regulargrid_kwargs,
            model_kwargs=dict( # for UNet
                in_chans=LEN_ALL_VARS + 1, # + 1 for flow_time
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
        return_intermediates=True,
        method="midpoint",  # 'midpoint' or 'euler'
        nlev=nlev,
        step_size=0.1,
    ),
)

flow = MODELWRAPPERS["flowmatching"](**wrapper_kwargs)

lit_module_kwargs = dict(
    model=flow,
    model_kwargs=wrapper_kwargs["model_kwargs"],
    loss="mse",
    loss_kwargs=dict(
        weights=LOSS_WEIGHTS, spectral_power_weight=0.0, nlat=len(lat), nlon=len(lon), normalize_batch=True,
    ),
    metrics=[
        dict(name=m, kwargs=dict(weights=METRIC_WEIGHTS))
        for m in ["rmse", "r2", "nse", "rabsbias", "rrmse"]
    ],
    # + [dict(name="mass_rmsev2", kwargs=dict(molecule=m)) for m in ["co2"]],
    no_grad_step_shedule=None,
    lr=1e-3,
    weight_decay=0.1,
    lr_shedule_kwargs=dict(
        warmup_steps=1000, halfcosine_steps=99000, min_lr=3e-7, max_lr=1.0
    ),
    val_dataloader_names=["singlestep", "rollout"],
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
    target_vars=["co2massmix"], #, "airmass"
    forcing_vars=[
        # "gph_bottom",
        # "gph_top",
        # "p_bottom",
        # "p_top",
        # "q",
        # "t",
        # "u",
        # "v",
        # "blh",
        # "cell_area",
        # "co2flux_anthro",
        # "co2flux_land",
        # "co2flux_ocean",
        # "orography",
        # "tisr",
    ],
    compute=False,
)

data_path_forecast = Path(
    "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/"
)


trainer_kwargs = dict(
    max_steps=10000,
    accelerator="gpu",
    devices=N_GPUS,
    log_every_n_steps=100,
    gradient_clip_val=32,
    precision="bf16-mixed",
    strategy=(
        "auto"
        if N_GPUS == 1
        else pl.strategies.DDPStrategy(find_unused_parameters=False)
    ), # profiler="simple", fast_dev_run=True,
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

obs_compare_path = f"/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/obs_carbontracker_{grid}_{vertical_levels}_{freq}.zarr"  # noqa: E501


def main(rollout: bool = False, train: bool = True, ckpt: str = "last", data_path: str|None = None) -> None:
    """Main function to run the training or rollout evaluation."""
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
            massfixers=["scale"],  # [None, "scale"],
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

# execute via:
# CUDA_VISIBLE_DEVICES=7 python3 -u /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_firstrun_20250730_dev/train.py
# or:
# sbatch train.slurm (check: squeue -u jgross)
