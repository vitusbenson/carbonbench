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


def plot_massconserve(preds, targs, preds_no_fix=None):
    preds["time"] = targs["time"]
    if preds_no_fix is not None:
        preds_no_fix["time"] = targs["time"]
    preds = preds.sel(time=slice("2018-01-01", "2019-01-01"))
    targs = targs.sel(time=slice("2018-01-01", "2019-01-01"))
    preds["level"] = targs["level"]

    targ_mass = (targs.co2massmix * targs.airmass) / 1e6
    pred_mass = (preds.co2massmix * targs.airmass) / 1e6

    sum_targ_mass = targ_mass.sum(["lat", "lon"]).compute() / 3.664
    sum_pred_mass = pred_mass.sum(["lat", "lon"]).compute() / 3.664

    total_targ_mass = sum_targ_mass.sum("level")
    total_pred_mass = sum_pred_mass.sum("level")

    diff_mass = sum_pred_mass - sum_targ_mass

    mass_rmse = (
        (total_targ_mass - total_pred_mass) ** 2
    ).mean().compute().item() ** 0.5
    relmass_rmse = (
        ((total_targ_mass - total_pred_mass) / total_targ_mass) ** 2
    ).mean().compute().item() ** 0.5

    if preds_no_fix is not None:
        preds_no_fix = preds_no_fix.sel(time=slice("2018-01-01", "2019-01-01"))
        preds_no_fix["level"] = targs["level"]
        pred_mass_no_fix = (preds_no_fix.co2massmix * targs.airmass) / 1e6
        sum_pred_mass_no_fix = pred_mass_no_fix.sum(["lat", "lon"]).compute() / 3.664
        total_pred_mass_no_fix = sum_pred_mass_no_fix.sum("level")
        mass_rmse_no_fix = (
            (total_targ_mass - total_pred_mass_no_fix) ** 2
        ).mean().compute().item() ** 0.5
        relmass_rmse_no_fix = (
            ((total_targ_mass - total_pred_mass_no_fix) / total_targ_mass) ** 2
        ).mean().compute().item() ** 0.5

    cum_targ_fluxes = (
        (
            (
                (
                    targs[["co2flux_land", "co2flux_ocean", "co2flux_anthro"]]
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
    with mpl.rc_context(mpl_rc_params):
        fig, axs = plt.subplots(1, 2, figsize=(0.9 * 12, 0.9 * 4))
        ax = axs[0]
        total_targ_mass.plot(
            x="time",
            color="tab:gray",
            ax=ax,
            label="Target Atmosphere",
            lw=1.6,
            zorder=2.2,
        )
        total_pred_mass.plot(
            x="time",
            color="tab:red",
            lw=0.8,
            ax=ax,
            label="Predicted Atmosphere",
            zorder=2.4,
        )
        if preds_no_fix is not None:
            total_pred_mass_no_fix.plot(
                x="time",
                color="tab:purple",
                lw=0.8,
                ax=ax,
                label="Predicted w/o Massfixer",
                zorder=2.4,
            )

        cum_targ_fluxes["co2flux_land"].plot(
            color="tab:green", ax=ax, label="Land Flux"
        )
        cum_targ_fluxes["co2flux_ocean"].plot(
            color="tab:blue", ax=ax, label="Ocean Flux"
        )
        cum_targ_fluxes["co2flux_anthro"].plot(
            color="tab:brown", ax=ax, label="Anthropogenic Flux"
        )
        (
            cum_targ_fluxes["co2flux_land"]
            + cum_targ_fluxes["co2flux_ocean"]
            + cum_targ_fluxes["co2flux_anthro"]
            - 2 * total_targ_mass.isel(time=0).item()
        ).plot(
            color="tab:orange",
            ax=ax,
            label="Flux Sum",
            lw=1.6,
            ls=(0, (5, 5)),
            zorder=2.3,
        )

        ax.set_ylabel(f"Total Mass [PgC]")
        ax.set_xlabel("")
        ax.set_ylim(
            850,  # cum_targ_fluxes.to_array("band").min().round(0).item() - 5,
            875,  # cum_targ_fluxes.to_array("band").max().round(0).item() + 2,
        )
        ax.text(
            0.05,
            0.83 if preds_no_fix is not None else 0.9,
            (
                f"W/ Mass fixer\nRMSE:        {mass_rmse:.5f} PgC\nRel. RMSE: {(relmass_rmse*100):.5f} %\nW/o Mass fixer\nRMSE:        {mass_rmse_no_fix:.5f} PgC\nRel. RMSE: {(relmass_rmse_no_fix*100):.5f} %"
                if preds_no_fix is not None
                else f"RMSE:        {mass_rmse:.5f} PgC\nRel. RMSE: {(relmass_rmse*100):.5f} %"
            ),
            fontsize=8,
            verticalalignment="center",
            horizontalalignment="left",
            transform=ax.transAxes,
        )

        for date in pd.date_range(
            start=targs.time[0].item(), end=targs.time[-1].item(), freq="QS"
        ):
            ax.axvline(x=date, color="grey", alpha=0.5, ls="--", lw=0.5, zorder=0)

        ax.legend(
            loc="lower left", bbox_to_anchor=(0, 0), frameon=False, title="", ncols=1
        )

        old_level = diff_mass.level
        diff_mass["level"] = range(len(old_level))

        diff_mass.plot(
            x="time",
            y="level",
            cbar_kwargs={"label": "Mass error [PgC]"},
            ax=axs[1],
            vmin=-0.125,
            vmax=0.125,
            cmap="RdBu_r",
        )
        axs[1].set_ylabel(f"Hybrid Level [hPa]")
        axs[1].set_xlabel("")
        axs[1].set_yticks(
            ticks=range(len(old_level)), labels=old_level.values.astype("int")
        )

    return fig


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

    if "massfixer=scale" in pred_path:
        pred_path_no_fix = pred_path.replace("massfixer=scale", "massfixer=None")
        pred_path_no_fix = Path(pred_path_no_fix)
        if pred_path_no_fix.exists():
            _, preds_no_fix = load_pred_targ(target_path, pred_path_no_fix)
            preds_no_fix = preds_no_fix.isel(time=slice(1, None))
        else:
            preds_no_fix = None
    else:
        preds_no_fix = None

    pred_path = Path(pred_path)
    out_dir = Path(out_dir) / "massconservation"
    out_dir.mkdir(exist_ok=True, parents=True)

    model_name = model_name or pred_path.parent.parent.parent.parent.parent.name
    experiment_name = experiment_name or pred_path.parent.parent.parent.parent.name
    singlestep_or_rollout = singlestep_or_rollout or pred_path.parent.parent.parent.name
    ckpt_name = ckpt_name or pred_path.parent.name

    targs, preds = load_pred_targ(target_path, pred_path)

    preds = preds.isel(time=slice(1, None))
    targs = targs.isel(time=slice(1, None)).isel(time=slice(len(preds.time)))

    fig = plot_massconserve(preds, targs, preds_no_fix=preds_no_fix)
    for imgformat in imgformats:
        plt.savefig(
            out_dir
            / f"{model_name}_{experiment_name}_{singlestep_or_rollout}_{ckpt_name}.{imgformat}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    plot_massconserve_per_model(
        args.target_path, args.pred_path, args.out_dir, imgformats=["pdf"]
    )
