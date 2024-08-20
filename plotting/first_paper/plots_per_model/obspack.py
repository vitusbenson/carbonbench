from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from neural_transport.tools.conversion import *

mpl_rc_params = {
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
}

ALL_OBSPACK_TYPES = [
    "surface-insitu",
    "aircraft-pfp",
    "aircraft-insitu",
    "surface-flask",
    "shipboard-insitu",
    "aircraft-flask",
    "aircore",
    "surface-pfp",
    "tower-insitu",
    "shipboard-flask",
]


def plot_obspack_stations(
    pred_path,
    obs,
    metadata,
    out_dir,
    compare_obs=None,
    ids=None,
    stations=["zep", "cba", "mlo", "asc", "crz", "psa"],
    ylims=[[390., 450.],[390., 450.],[400., 420.],[400., 420.],[400., 415.],[400., 415.]],
    types=ALL_OBSPACK_TYPES,
    quality=["representative"],
    levels="default",
    freq="QS",
    imgformats=["svg", "png", "pdf"],
    model_name=None,
    experiment_name=None,
    singlestep_or_rollout=None,
    ckpt_name=None,
):
    pred_path = Path(pred_path)
    out_dir = Path(out_dir) / "obspack"
    out_dir.mkdir(exist_ok=True, parents=True)

    model_name = model_name or pred_path.parent.parent.parent.parent.parent.name
    experiment_name = experiment_name or pred_path.parent.parent.parent.parent.name
    singlestep_or_rollout = singlestep_or_rollout or pred_path.parent.parent.parent.name
    ckpt_name = ckpt_name or pred_path.parent.name

    out_dir.mkdir(exist_ok=True, parents=True)

    if "co2molemix" not in obs:
        obs["co2molemix"] = massmix_to_molemix(obs.co2massmix)
    if compare_obs is not None:
        if "co2molemix" not in compare_obs:
            if "co2massmix" not in compare_obs:
                compare_obs["co2massmix"] = density_to_massmix(
                    compare_obs.co2density, compare_obs.airdensity, ppm=True
                )
            compare_obs["co2molemix"] = massmix_to_molemix(compare_obs.co2massmix)

    metadata["default_level"] = metadata.level == metadata.groupby(
        ["station", "quality", "type"]
    )["level"].transform("max")
    if stations == "all":
        stations = metadata.station.unique()
    if ids is not None:
        subset = metadata[metadata.id.isin(ids)]
    else:
        subset = metadata[
            (metadata.station.isin(stations))
            & (metadata.type.isin(types))
            & (metadata.quality.isin(quality))
        ]
        if levels == "default":
            subset = subset[subset.default_level]
        elif isinstance(levels, list):
            subset = subset[subset.level.isin(levels)]

    filenames = pd.Series(obs.obs_filename.max("time"))

    with mpl.rc_context(mpl_rc_params):
        fig, axs = plt.subplots(
            len(stations) // 2, 2, figsize=(8, 5), sharex=True, sharey="row"
        )
        for ax_idx, (ax, station, ylim) in enumerate(zip(axs.flat, stations, ylims)):
            row = subset[subset.station == station].iloc[0]
            try:
                i = np.where(filenames == row["id"])[0][0]
            except:
                continue

            obs.obs_co2molemix.isel(cell=i).plot(
                ax=ax, color="black", lw=0.75, alpha=0.85, label="Observed", marker="x"
            )

            if compare_obs is not None:
                compare_obs.co2molemix.isel(cell=i).plot(
                    ax=ax, label="Inversion", color="tab:green", lw=0.75
                )

            obs.co2molemix.isel(cell=i).plot(
                ax=ax, label="Predicted", color="tab:orange", lw=0.75
            )

            ax.set_xlabel("")
            ax.set_title(f"{row['site_name']}, Level {row['level']}")
            ax.set_ylim(*ylim)

            ax.set_ylabel("CO2 molemix [ppm]" if ax_idx % 2 == 0 else "")

            for date in pd.date_range(
                start=obs.time[0].item(), end=obs.time[-1].item(), freq=freq
            ):
                ax.axvline(x=date, color="grey", alpha=0.5, ls="--", lw=0.5, zorder=0)
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(
        #     handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3
        # )
        plt.legend()
        sns.move_legend(
            axs[-1, 1],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            frameon=False,
            # title="Pressure Level [hPa]",
        )
        plt.tight_layout()

        for imgformat in imgformats:
            plt.savefig(
                out_dir
                / f"{model_name}_{experiment_name}_{singlestep_or_rollout}_{ckpt_name}.{imgformat}",
                dpi=300,
            )
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--compare_path", type=str, required=True)
    parser.add_argument("--obspred_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    obspreds = xr.open_zarr(args.obspred_path)

    obspack_metadata = pd.read_csv(args.metadata_path)

    compare_obspred = xr.open_zarr(args.compare_path)

    plot_obspack_stations(
        args.obspred_path,
        obspreds,
        obspack_metadata,
        args.out_dir,
        compare_obs=compare_obspred,
        imgformats=["pdf"],
    )
