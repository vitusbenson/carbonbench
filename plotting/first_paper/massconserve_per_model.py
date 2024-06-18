# %%
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import xrft

from neural_transport.inference.plot_results import get_pred_targ_from_varname
from neural_transport.tools.conversion import *
from neural_transport.training.train import load_pred_targ

mpl_rc_params = {
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
}


# %%
# pred_path = "/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240309/sfno_L_addflux/lightning_logs/version_1/preds/last/co2_pred_rollout_QS.zarr"
# target_path = (
#     "/User/homes/vbenson/vbenson/graph_tm/data/Carboscope/test/carboscope_latlon4.zarr"
# )

# targs, preds = load_pred_targ(target_path, pred_path)


# %%
def plot_massconserve(preds, targs):
    preds["time"] = targs["time"]
    preds = preds.sel(time=slice("2018-01-01", "2019-01-01"))
    targs = targs.sel(time=slice("2018-01-01", "2019-01-01"))
    preds["height"] = targs["height"]

    targs["co2massmix"] = density_to_massmix(
        targs.co2density, targs.airdensity, ppm=True
    )
    targ_mass = 2.124 * massmix_to_molemix(targs.co2massmix)
    pred_mass = 2.124 * massmix_to_molemix(preds.co2massmix)

    lat_weights = np.cos(np.radians(targs.lat))
    lat_weights = lat_weights / lat_weights.sum()
    pressure_height = -np.diff(np.concatenate([targs.height, [0.0]]))
    height_weight = xr.DataArray(
        pressure_height / pressure_height.sum(),
        coords={"height": targs.height},
        dims=("height",),
    )

    sum_targ_mass = (targ_mass * lat_weights).sum("lat").mean("lon").compute()
    sum_pred_mass = (pred_mass * lat_weights).sum("lat").mean("lon").compute()

    total_targ_mass = (sum_targ_mass * height_weight).sum("height")
    total_pred_mass = (sum_pred_mass * height_weight).sum("height")

    # targ_mass = density_to_mass(targs.co2density, targs.volume)
    # pred_mass = density_to_mass(preds.co2density, targs.volume)

    # sum_targ_mass = targ_mass.sum(["lat", "lon"]).compute()
    # sum_pred_mass = pred_mass.sum(["lat", "lon"]).compute()

    diff_mass = sum_pred_mass - sum_targ_mass

    cum_targ_fluxes = (
        (
            (
                (
                    targs[["co2flux_land", "co2flux_ocean", "co2flux_subt"]]
                    * targs.cell_area
                    * 60
                    * 60
                    * 6
                )
                / 1e12
                / 3.664
            )
            .sum(["lat", "lon"])
            .cumsum("time")
            + total_targ_mass.isel(time=0).item()
        )
        .compute()
        .assign_coords(time=targs.time)
    )

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    ax = axs[0]
    total_targ_mass.plot(
        x="time", color="tab:gray", ax=ax, label="Target Atmosphere", lw=1.6, zorder=2.2
    )
    total_pred_mass.plot(
        x="time",
        color="tab:red",
        lw=0.8,
        ax=ax,
        label="Predicted Atmosphere",
        zorder=2.4,
    )
    cum_targ_fluxes["co2flux_land"].plot(color="tab:green", ax=ax, label="Land Flux")
    cum_targ_fluxes["co2flux_ocean"].plot(color="tab:blue", ax=ax, label="Ocean Flux")
    cum_targ_fluxes["co2flux_subt"].plot(
        color="tab:brown", ax=ax, label="Anthropogenic Flux"
    )
    (
        cum_targ_fluxes["co2flux_land"]
        + cum_targ_fluxes["co2flux_ocean"]
        + cum_targ_fluxes["co2flux_subt"]
        - 2 * total_targ_mass.isel(time=0).item()
    ).plot(
        color="tab:orange", ax=ax, label="Flux Sum", lw=1.6, ls=(0, (5, 5)), zorder=2.3
    )

    ax.set_ylabel(f"Total Mass [PgC]")
    ax.set_xlabel("")
    ax.set_ylim(
        850,  # cum_targ_fluxes.to_array("band").min().round(0).item() - 5,
        875,  # cum_targ_fluxes.to_array("band").max().round(0).item() + 2,
    )

    for date in pd.date_range(
        start=targs.time[0].item(), end=targs.time[-1].item(), freq="QS"
    ):
        ax.axvline(x=date, color="grey", alpha=0.5, ls="--", lw=0.5, zorder=0)

    ax.legend(loc="lower left", bbox_to_anchor=(0, 0), frameon=False, title="", ncols=1)

    (diff_mass * height_weight).plot(
        x="time",
        y="height",
        cbar_kwargs={"label": "Mass error [PgC]"},
        ax=axs[1],
        vmin=-0.25,
        vmax=0.25,
        cmap="RdBu_r",
    )
    axs[1].set_ylabel(f"Pressure Level [hPa]")
    axs[1].set_xlabel("")
    # sns.set_palette("Spectral", n_colors=19)
    # (diff_mass / 1e12).plot(x="time", hue="height", ax=axs[1])
    # # (diff_mass.sum("height") / 1e12).plot(x = "time", color = "black", ax = ax)
    # axs[1].set_ylabel(f"Mass error [PgCO$_2$]")
    # axs[1].set_xlabel("")
    # sns.move_legend(
    #     axs[1],
    #     loc="upper left",
    #     bbox_to_anchor=(1, 1),
    #     frameon=False,
    #     title="Pressure Level [hPa]",
    # )

    return fig


# %%
def plot_massconserve_per_model(
    target_path,
    pred_path,
    out_dir,
    imgformats=["svg", "png", "pdf"],
    model_name=None,
    experiment_name=None,
    singlestep_or_rollout=None,
    ckpt_name=None,
):
    pred_path = Path(pred_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    model_name = model_name or pred_path.parent.parent.parent.parent.parent.name
    experiment_name = experiment_name or pred_path.parent.parent.parent.parent.name
    singlestep_or_rollout = singlestep_or_rollout or pred_path.parent.parent.parent.name
    ckpt_name = ckpt_name or pred_path.parent.name

    targs, preds = load_pred_targ(target_path, pred_path)

    preds = preds.isel(time=slice(1, None))
    targs = targs.isel(time=slice(1, None)).isel(time=slice(len(preds.time)))

    fig = plot_massconserve(preds, targs)
    for imgformat in imgformats:
        plt.savefig(
            out_dir
            / f"massconserve_{model_name}_{experiment_name}_{singlestep_or_rollout}_{ckpt_name}.{imgformat}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()

    return None


if __name__ == "__main__":

    if True:

        experiments = [
            dict(
                pred_path="/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carboscope/sfno/sfnov2_L/singlestep/preds/best/co2_pred_rollout_QS.zarr",  # "/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240309/sfno_L_addflux/lightning_logs/version_1/preds/last/co2_pred_rollout_QS.zarr",
                model_name="sfno",
                experiment_name="sfno_L",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240115/graphcast_L0-L3_M/lightning_logs/version_1/preds/last/co2_pred_rollout_QS.zarr",
                model_name="graphcast",
                experiment_name="graphcast_L0-L3_M",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240404/graphtm_L3_M/lightning_logs/version_1/preds/last/co2_pred_rollout_QS.zarr",
                model_name="icosagnn",
                experiment_name="icosagnn_L3_M",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240522/hybridsfno_L/lightning_logs/version_1/preds/last/co2_pred_rollout_QS.zarr",
                model_name="hybridsfno",
                experiment_name="hybridsfno_L",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240517/swintransformer_M/lightning_logs/version_1/preds/last/co2_pred_rollout_QS.zarr",
                model_name="swintransformer",
                experiment_name="swintransformer_M",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
        ]

        target_path = "/User/homes/vbenson/vbenson/graph_tm/data/Carboscope/test/carboscope_latlon4.zarr"
        out_dir = "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/massconserve_per_model"
        from tqdm import tqdm

        for experiment in tqdm(experiments):
            print(experiment["model_name"])
            plot_massconserve_per_model(
                target_path,
                experiment["pred_path"],
                out_dir,
                model_name=experiment["model_name"],
                experiment_name=experiment["experiment_name"],
                singlestep_or_rollout=experiment["singlestep_or_rollout"],
                ckpt_name=experiment["ckpt_name"],
            )
