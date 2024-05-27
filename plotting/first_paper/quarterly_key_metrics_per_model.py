from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from neural_transport.inference.analyse import freq_mean
from neural_transport.inference.plot_results import (
    METRIC_LABELS,
    METRICS,
    get_metric_limits,
    get_pred_targ_from_varname,
)
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


def plot_four_metrics_over_leadtime(preds, targs, varname, figsize=(11, 5), freq="QS"):

    pred, targ = get_pred_targ_from_varname(preds, targs, varname)

    metrics = ["rmse", "r2", "rel_mean", "rel_std"]

    weights = np.cos(np.deg2rad(targ.lat.compute()))
    _, weights = xr.broadcast(targ, weights)

    with mpl.rc_context(mpl_rc_params):
        sns.set_palette("Spectral", n_colors=19)
        fig, axs = plt.subplots(
            2, 2, figsize=figsize, sharex=True, gridspec_kw=dict(hspace=0.05)
        )
        for i, (ax, metric) in enumerate(zip(axs.flat, metrics)):
            metric_func = METRICS[metric]

            metric_pred = metric_func(pred.compute(), targ.compute(), weights)
            metric_pred["height"] = metric_pred["height"].values.round(0).astype("int")

            ylim = get_metric_limits(metric, metric_pred)

            metricf = freq_mean(metric_pred, freq=freq).rename(time="days")

            metricf.plot(hue="height", ax=ax, add_legend=(i == 1))

            ax.set_ylim(*ylim)
            ax.set_xticks([0, 20, 40, 60, 80])
            ax.set_xlabel("" if i < 2 else "Lead time [days]")
            ax.set_ylabel("RMSE [ppm]" if metric == "rmse" else METRIC_LABELS[metric])

            if metric == "r2":
                thresh_value = 0.8
                ax.axhline(y=thresh_value, ls="--", color="black", zorder=0)

                invalid_days = (
                    metricf.min("height")
                    .compute()
                    .where(lambda x: x < thresh_value, drop=True)
                    .days.values
                )
            elif metric == "rmse":
                thresh_value = 1.0
                ax.axhline(y=thresh_value, ls="--", color="black", zorder=0)

                invalid_days = (
                    metricf.max("height")
                    .compute()
                    .where(lambda x: x > thresh_value, drop=True)
                    .days.values
                )
            else:
                ax.axhline(y=0.999, ls="--", color="black", zorder=0)
                ax.axhline(y=1.001, ls="--", color="black", zorder=0)
                invalid_days = (
                    (np.abs(metricf - 1))
                    .max("height")
                    .compute()
                    .where(lambda x: x > 0.001, drop=True)
                    .days.values
                )

            min_days = (
                metricf.days.values[-1] if len(invalid_days) == 0 else invalid_days[0]
            )
            if min_days < 90:
                ax.axvline(x=min_days, color="black", zorder=0, lw=0.5)
                trans = ax.get_xaxis_transform()
                if min_days < 60:
                    ax.text(
                        min_days + 1,
                        0.05 if metric == "r2" else 0.95,
                        f"{min_days} days stable",
                        transform=trans,
                        fontsize=8,
                        verticalalignment="center",
                    )
                else:
                    ax.text(
                        min_days - 1,
                        0.05 if metric == "r2" else 0.95,
                        f"{min_days} days stable",
                        transform=trans,
                        fontsize=8,
                        verticalalignment="center",
                        horizontalalignment="right",
                    )
            else:
                ax.text(
                    89,
                    0.05 if metric == "r2" else 0.95,
                    f">90 days stable",
                    transform=trans,
                    fontsize=8,
                    verticalalignment="center",
                    horizontalalignment="right",
                )

        sns.move_legend(
            axs[0, 1],
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            title="Pressure Level [hPa]",
        )
        fig.tight_layout()

    return fig


def plot_quarterly_key_metrics_for_model(
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

    fig = plot_four_metrics_over_leadtime(preds, targs, "co2molemix", freq="QS")
    for imgformat in imgformats:
        plt.savefig(
            out_dir
            / f"keymetrics_over_leadtime_{model_name}_{experiment_name}_{singlestep_or_rollout}_{ckpt_name}.{imgformat}",
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

    plot_quarterly_key_metrics_for_model(args.target_path, args.pred_path, args.out_dir)

    if False:

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
        out_dir = "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/quarterly_key_metrics_per_model"
        from tqdm import tqdm

        for experiment in tqdm(experiments):
            print(experiment["model_name"])
            plot_quarterly_key_metrics_for_model(
                target_path,
                experiment["pred_path"],
                out_dir,
                model_name=experiment["model_name"],
                experiment_name=experiment["experiment_name"],
                singlestep_or_rollout=experiment["singlestep_or_rollout"],
                ckpt_name=experiment["ckpt_name"],
            )
