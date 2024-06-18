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
    obs,
    metadata,
    out_dir,
    model,
    compare_obs=None,
    ids=None,
    stations=["zep", "cba", "mlo", "asc", "crz", "psa"],
    types=ALL_OBSPACK_TYPES,
    quality=["representative"],
    levels="default",
    freq="QS",
    imgformats=["svg", "png", "pdf"],
):
    out_dir = Path(out_dir)
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
        for ax_idx, (ax, station) in enumerate(zip(axs.flat, stations)):
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
            plt.savefig(out_dir / f"obspack_{model}.{imgformat}", dpi=300)
        plt.close()


if __name__ == "__main__":

    if True:

        experiments = [
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240309/sfno_L_addflux/lightning_logs/version_1/preds/last/obs_co2_pred_rollout_QS.zarr",
                model_name="sfno",
                experiment_name="sfno_L",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240115/graphcast_L0-L3_M/lightning_logs/version_1/preds/last/obs_co2_pred_rollout_QS.zarr",
                model_name="graphcast",
                experiment_name="graphcast_L0-L3_M",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240404/graphtm_L3_M/lightning_logs/version_1/preds/last/obs_co2_pred_rollout_QS.zarr",
                model_name="icosagnn",
                experiment_name="icosagnn_L3_M",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240522/hybridsfno_L/lightning_logs/version_1/preds/last/obs_co2_pred_rollout_QS.zarr",
                model_name="hybridsfno",
                experiment_name="hybridsfno_L",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240517/swintransformer_M/lightning_logs/version_1/preds/last/obs_co2_pred_rollout_QS.zarr",
                model_name="swintransformer",
                experiment_name="swintransformer_M",
                singlestep_or_rollout="rollout",
                ckpt_name="last",
            ),
        ]

        target_path = "/User/homes/vbenson/vbenson/graph_tm/data/Carboscope/test/carboscope_latlon4.zarr"
        out_dir = "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/timeseries_per_model"
        from tqdm import tqdm

        for experiment in tqdm(experiments):
            print(experiment["model_name"])
            obspreds = xr.open_zarr(experiment["pred_path"])

            obspack_metadata = pd.read_csv(
                "/User/homes/vbenson/vbenson/graph_tm/data/Obspack/obspack_metadata.csv"
            )

            carboscope_obspred = xr.open_zarr(
                "/User/homes/vbenson/vbenson/graph_tm/data/Carboscope/test/obspack_carboscope.zarr"
            )

            plot_obspack_stations(
                obspreds,
                obspack_metadata,
                out_dir,
                model=experiment["model_name"],
                compare_obs=carboscope_obspred,
            )
