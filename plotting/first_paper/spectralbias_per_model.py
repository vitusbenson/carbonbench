from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xrft

from neural_transport.inference.plot_results import get_pred_targ_from_varname
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


def get_zonal_spectrum(arr):
    arr = arr.compute()

    Farr = xrft.fft(arr, dim="lon", real_dim="lon")

    Specarr = abs(Farr).mean(["lat", "height"])

    return Specarr


def plot_zonal_spectrum_line(pred, targ, figsize=(8, 5), **kwargs):

    Specpred = get_zonal_spectrum(pred)
    Spectarg = get_zonal_spectrum(targ)

    pred_start_dates = pd.date_range(
        targ.time[0].values, targ.time[-1].values, freq="QS", inclusive="left"
    ) - pd.Timedelta("6h")

    with mpl.rc_context(mpl_rc_params):
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot()

        cmap = mpl.colormaps.get_cmap("plasma")

        for days_ahead, color, time in zip(
            [
                0.25,
                1,
                7,
                30,
                60,
            ],
            [
                cmap(v) for v in np.linspace(0.25, 0.8, 5)
            ],  # ["tab:blue", "tab:orange", "tab:green"],
            [
                "6h",
                "1d",
                "7d",
                "30d",
                "60d",
            ],
        ):

            Specpred.sel(
                time=(pred_start_dates + pd.Timedelta(f"{24*days_ahead}h"))
            ).mean("time").plot(
                yscale="log",
                ax=ax,
                color=color,
                ls="--",
                label=f"Prediction {time} ahead",
            )

        Spectarg.mean("time").plot(
            yscale="log", ax=ax, color="black", ls="-", label=f"Mean Ground Truth"
        )

        ax.set_title("Zonal Power Spectrum")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.set_ylim(1e0, 1e3)
        plt.legend()

        plt.tight_layout()

    return fig


def plot_spectralbias_per_model(
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

    model_name = model_name or pred_path.parent.parent.parent.parent.parent.name
    experiment_name = experiment_name or pred_path.parent.parent.parent.parent.name
    singlestep_or_rollout = singlestep_or_rollout or pred_path.parent.parent.parent.name
    ckpt_name = ckpt_name or pred_path.parent.name

    targs, preds = load_pred_targ(target_path, pred_path)

    preds = preds.isel(time=slice(1, None))
    targs = targs.isel(time=slice(1, None)).isel(time=slice(len(preds.time)))

    pred, targ = get_pred_targ_from_varname(
        preds,
        targs,
        "co2molemix",
    )
    fig = plot_zonal_spectrum_line(pred, targ)
    for imgformat in imgformats:
        plt.savefig(
            out_dir
            / f"zonalspectrum_{model_name}_{experiment_name}_{singlestep_or_rollout}_{ckpt_name}.{imgformat}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()

    return None


if __name__ == "__main__":

    if True:

        experiments = [
            dict(
                pred_path="/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240309/sfno_L_addflux/lightning_logs/version_1/preds/last/co2_pred_rollout_QS.zarr",
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
        out_dir = "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/spectrum_per_model"
        from tqdm import tqdm

        for experiment in tqdm(experiments[::-1]):
            print(experiment["model_name"])
            plot_spectralbias_per_model(
                target_path,
                experiment["pred_path"],
                out_dir,
                model_name=experiment["model_name"],
                experiment_name=experiment["experiment_name"],
                singlestep_or_rollout=experiment["singlestep_or_rollout"],
                ckpt_name=experiment["ckpt_name"],
            )
