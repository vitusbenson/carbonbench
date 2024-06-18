from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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


def plot_rmse(preds, targs, varname):

    unit = targs.attrs.get("units", "")
    long_name = targs.attrs.get("long_name", varname)
    clabel = f"{long_name} [{unit}]"
    clabel_delta = f"Delta [{unit}]"

    pred, targ = get_pred_targ_from_varname(
        preds,
        targs,
        varname,
    )

    se = ((pred - targ) ** 2).compute()

    rmse_map = se.mean(["height", "time"]) ** 0.5
    rmse_latheight = se.mean(["time", "lon"]) ** 0.5

    vmin = 0  # min(
    #     rmse_map.quantile(0.02).compute().item(),
    #     rmse_latheight.quantile(0.02).compute().item(),
    # )
    vmax = max(
        rmse_map.quantile(0.98).compute().item(),
        rmse_latheight.quantile(0.98).compute().item(),
    )

    nstep = 101

    kwargs = dict(
        clabel=clabel,
        clabel_delta=clabel_delta,
        vmin=vmin,
        vmax=vmax,
        nstep=nstep,
    )

    with mpl.rc_context(mpl_rc_params):
        fig = plt.figure(figsize=(18.03 * 0.8, 6 * 0.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.82, 1])
        ax1 = fig.add_subplot(gs[0], projection=ccrs.Robinson())

        levels = np.linspace(
            kwargs.get("vmin", vmin),
            kwargs.get("vmax", vmax),
            kwargs.get("nstep", 21),
        )
        cmap = plt.get_cmap("PuRd", len(levels))

        rmse_map.plot(
            ax=ax1,
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            levels=levels,
            transform=ccrs.PlateCarree(),
            zorder=0,
        )
        ax1.set_global()

        gl = ax1.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="dimgray",
            alpha=0.4,
            zorder=2,
        )
        gl.xlabel_style = {"size": 6, "color": "dimgray"}
        gl.ylabel_style = {"size": 6, "color": "dimgray"}

        gl.right_labels = False
        gl.geo_labels = True
        gl.bottom_labels = True

        gl.top_labels = False
        gl.left_labels = True

        ax1.coastlines(linewidth=0.5, zorder=2)

        ax2 = fig.add_subplot(gs[1])

        rmse_latheight.plot(
            y="lat",
            x="height",
            ax=ax2,
            add_colorbar=True,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            levels=levels,
            zorder=0,
            cbar_kwargs=dict(label="RMSE [ppm]"),
        )
        ax2.set_xlabel(f"Pressure Level [hPa]")
        ax2.set_ylabel("Latitude")
        fig.tight_layout()

    return fig


def plot_rmse_per_model(
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

    fig = plot_rmse(preds, targs, "co2molemix")
    for imgformat in imgformats:
        plt.savefig(
            out_dir
            / f"rmse_{model_name}_{experiment_name}_{singlestep_or_rollout}_{ckpt_name}.{imgformat}",
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
        out_dir = "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/rmse_per_model"
        from tqdm import tqdm

        for experiment in tqdm(experiments[::-1]):
            print(experiment["model_name"])
            plot_rmse_per_model(
                target_path,
                experiment["pred_path"],
                out_dir,
                model_name=experiment["model_name"],
                experiment_name=experiment["experiment_name"],
                singlestep_or_rollout=experiment["singlestep_or_rollout"],
                ckpt_name=experiment["ckpt_name"],
            )
