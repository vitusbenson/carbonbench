# %%
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from neural_transport.inference.analyse import compute_local_scores

mpl_rc_params = {
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
}


def get_obspack_tm3_dataframe(pred_path):
    ds = xr.open_zarr(pred_path)
    df = compute_local_scores(ds)
    return df


def collect_dataframes_from_runs(runs):
    sort_order = {m[0]: i for i, m in enumerate(runs)}
    global_dfs = []
    for model, res, path in runs:
        if model == "TM5":
            continue
        df = pd.read_csv(
            path / "rollout" / "scores" / "ckpt=best_massfixer=scale" / "metrics.csv"
        )
        df["model"] = model
        df["resolution"] = res
        global_dfs.append(df)
    global_df = pd.concat(global_dfs)
    global_df["model"] = pd.Categorical(
        global_df["model"],
        categories=sorted(sort_order, key=sort_order.get),
        ordered=True,
    )
    global_df["resolution"] = pd.Categorical(
        global_df["resolution"],
        # categories=[
        #     i for i in ["LowRes", "MidRes", "OrigRes"] if i in global_df["resolution"]
        # ],
        ordered=True,
    )

    station_dfs = []
    for model, res, path in runs:
        if model == "TM5":
            df = get_obspack_tm3_dataframe(path)
        else:
            df = pd.read_csv(
                path
                / "rollout"
                / "scores"
                / "ckpt=best_massfixer=scale"
                / "obs_metrics.csv"
            )
        df["model"] = model
        df["resolution"] = res
        station_dfs.append(df)

    station_df = pd.concat(station_dfs)
    station_df["model"] = pd.Categorical(
        station_df["model"],
        categories=sorted(sort_order, key=sort_order.get),
        ordered=True,
    )
    station_df["resolution"] = pd.Categorical(
        station_df["resolution"],
        # categories=[
        #     i for i in ["LowRes", "MidRes", "OrigRes"] if i in station_df["resolution"]
        # ],
        ordered=True,
    )

    speed_dfs = []
    for model, res, path in runs:
        if model == "TM5":
            df = pd.DataFrame(
                dict(
                    LowRes=[[0.5 * 2, 0.75 * 2, 2]],
                    MidRes=[[0.5 * 8, 0.75 * 8, 8]],
                    OrigRes=[[0.5 * 8, 0.75 * 8, 8]],
                )[res],
                columns=["model_time", "read_time", "io_time"],
            )
        else:
            df = pd.read_csv(path / "benchmark_times.csv")
            df = pd.concat(
                [
                    df[~df.read_on_step & ~df.write_on_step]
                    .time.reset_index(drop=True)
                    .rename("model_time"),
                    df[df.read_on_step & ~df.write_on_step]
                    .time.reset_index(drop=True)
                    .rename("read_time"),
                    df[df.read_on_step & df.write_on_step]
                    .time.reset_index(drop=True)
                    .rename("io_time"),
                ],
                axis=1,
            )
        df["model"] = model
        df["resolution"] = res
        speed_dfs.append(df)
        # model_time_mean = df[~df.read_on_step & ~df.write_on_step].time.mean().item()
        # model_time_std = df[~df.read_on_step & ~df.write_on_step].time.std()
        # read_time_mean = df[df.read_on_step & ~df.write_on_step].time.mean().item()
        # read_time_std = df[df.read_on_step & ~df.write_on_step].time.std()
        # io_time_mean = df[df.read_on_step & df.write_on_step].time.mean().item()
        # io_time_std = df[df.read_on_step & df.write_on_step].time.std()

        # speed_dfs.append(dict(model = model, model_time_mean = model_time_mean, model_time_std = model_time_std, read_time_mean = read_time_mean, read_time_std = read_time_std, io_time_mean = io_time_mean, io_time_std = io_time_std))

    speed_df = pd.concat(speed_dfs)  # pd.DataFrame(speed_dfs)
    speed_df["model"] = pd.Categorical(
        speed_df["model"],
        categories=sorted(sort_order, key=sort_order.get),
        ordered=True,
    )
    speed_df["resolution"] = pd.Categorical(
        speed_df["resolution"],
        # categories=[
        #     i for i in ["LowRes", "MidRes", "OrigRes"] if i in speed_df["resolution"]
        # ],
        ordered=True,
    )

    return global_df, station_df, speed_df


def plot_model_intercomparison_barplots(
    global_df,
    station_df,
    speed_df,
    out_path,
    imgformats=["svg", "png", "pdf"],
    perf_metric="rmse",
):
    with mpl.rc_context(mpl_rc_params):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        station_hq_df = station_df[
            station_df.obs_filename.str.contains("representative")
        ]

        models = list(station_hq_df.model.cat.categories)
        colors = list(sns.color_palette("deep"))[: len(models)]
        resolutions = list(station_hq_df.resolution.cat.categories)
        markers = ["o", "s", "^", "p", "P", "v", "X", "D", "<", ">"][: len(models)]

        baseline_perf = station_hq_df[
            (station_hq_df.model == "TM5") & (station_hq_df.resolution == "OrigRes")
        ][perf_metric]
        baseline_time = speed_df[
            (speed_df.model == "TM5") & (speed_df.resolution == "OrigRes")
        ].model_time  # .io_time

        for res, marker in zip(resolutions, markers):
            for model, color in zip(models, colors):

                model_perf = station_hq_df[
                    (station_hq_df.model == model) & (station_hq_df.resolution == res)
                ][perf_metric]

                perf_mean = (1 - (model_perf / baseline_perf.mean()).mean()) * 100
                perf_std = (
                    2
                    * (model_perf / baseline_perf.mean()).std()
                    / np.sqrt(model_perf.count())
                ) * 100

                model_time = speed_df[
                    (speed_df.model == model) & (speed_df.resolution == res)
                ].model_time  # .io_time

                io_mean = (1 - (model_time / baseline_time.mean()).mean()) * 100
                io_std = (
                    2
                    * (model_time / baseline_time.mean()).std()
                    / np.sqrt(model_time.count())
                ) * 100

                pct_io = (
                    speed_df[
                        (speed_df.model == model) & (speed_df.resolution == res)
                    ].io_time.mean()
                    - speed_df[
                        (speed_df.model == model) & (speed_df.resolution == res)
                    ].model_time.mean()
                ) / speed_df[
                    (speed_df.model == model) & (speed_df.resolution == res)
                ].io_time.mean()

                perf_median = (1 - (model_perf / baseline_perf).median()) * 100
                perf_p25 = np.abs(
                    perf_median
                    - (1 - (model_perf / baseline_perf).quantile(0.25)) * 100
                )
                perf_p75 = (
                    np.abs(1 - (model_perf / baseline_perf).quantile(0.75)) * 100
                    - perf_median
                )

                io_median = (1 - (model_time / baseline_time).median()) * 100
                io_p25 = np.abs(
                    io_median - (1 - (model_time / baseline_time).quantile(0.25)) * 100
                )
                io_p75 = (
                    np.abs(1 - (model_time / baseline_time).quantile(0.75)) * 100
                    - io_median
                )

                ax.errorbar(
                    perf_mean,
                    io_mean,
                    xerr=perf_std,
                    yerr=io_std,
                    fmt=marker,
                    color=color,
                    label=model,
                )
                # ax.errorbar(
                #     perf_median,
                #     io_median,
                #     xerr=np.array([perf_p25, perf_p75])[:, None],
                #     yerr=np.array([io_p25, io_p75])[:, None],
                #     fmt=marker,
                #     color=color,
                #     label=model,
                # )
                # ax.annotate(
                #     f"IO={pct_io*100:.0f}%",
                #     (perf_mean * 1.05, io_mean * 1.05),
                #     fontsize=8,
                # )
        # ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
        # ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        max_pct = 100
        max_pct_rmse = 100  # max_pct
        ax.set_xlim(-max_pct_rmse, max_pct_rmse)
        ax.set_ylim(-max_pct, max_pct)
        ax.set_xlabel("RMSE Improvement [%]")  # r"$R^2$")
        ax.set_ylabel(
            "Runtime Improvement [%]"  # r"Runtime [$\frac{seconds}{Month}$]"
        )  # r"Speed [$\frac{Month}{second}$]")
        # legend
        ax.axhline(0, color="k", linestyle="--", lw=0.5, zorder=0.1)
        ax.axvline(0, color="k", linestyle="--", lw=0.5, zorder=0.1)
        ax.fill_between(
            [-max_pct_rmse, 0], -max_pct, 0, alpha=0.1, color="tab:red", zorder=0.05
        )  # blue
        ax.fill_between(
            [0, max_pct_rmse], -max_pct, 0, alpha=0.1, color="tab:orange", zorder=0.05
        )  # yellow
        ax.fill_between(
            [-max_pct_rmse, 0], 0, max_pct, alpha=0.1, color="tab:orange", zorder=0.05
        )  # orange
        ax.fill_between(
            [0, max_pct_rmse], 0, max_pct, alpha=0.1, color="tab:green", zorder=0.05
        )  # red

        ax.text(
            -(max_pct_rmse - 5),
            -(max_pct - 5),
            "slower & less accurate",
            fontsize=10,
            color="tab:red",
            ha="left",
            va="center",
        )
        ax.text(
            (max_pct_rmse - 5),
            -(max_pct - 5),
            "slower & more accurate",
            fontsize=10,
            color="tab:orange",
            ha="right",
            va="center",
        )
        ax.text(
            -(max_pct_rmse - 5),
            (max_pct - 5),
            "faster & less accurate",
            fontsize=10,
            color="tab:orange",
            ha="left",
            va="center",
        )
        ax.text(
            (max_pct_rmse - 5),
            (max_pct - 5),
            "faster & more accurate",
            fontsize=10,
            color="tab:green",
            ha="right",
            va="center",
        )

        # bbox_props = dict(boxstyle="larrow", fc="w", ec="k", lw=1)
        # t = ax.text(
        #     0.15,
        #     0.15,
        #     "   better   ",
        #     ha="center",
        #     va="center",
        #     rotation=45,
        #     fontsize=10,
        #     bbox=bbox_props,
        #     transform=ax.transAxes,
        # )

        rows = [mpatches.Patch(color=colors[i]) for i in range(len(colors))]
        columns = [
            plt.plot([], [], markers[i], markerfacecolor="k", markeredgecolor="k")[0]
            for i in range(len(markers))
        ]

        ax.legend(
            [mpatches.Patch(visible=False)]
            + rows
            + [mpatches.Patch(visible=False)]
            + columns,
            ["Model"] + models + ["Resolution"] + resolutions,
        )

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()

        for imgformat in imgformats:
            plt.savefig(
                Path(out_path) / f"speed_intercomparison_paretoplot.{imgformat}",
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()


# %%
if __name__ == "__main__":
    # %%
    Models = [
        # (
        #     "UNet",
        #     Path(
        #         "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker/unet/unet_M_targshift_latlon5.625_l10_6h/rollout/scores/ckpt=best_massfixer=scale"
        #     ),
        # ),
        (
            "UNet",  # "UNetAroma",
            "LowRes",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/unet/unet_M_tsaf_specloss/"
            ),
        ),
        (
            "GraphCast",
            "LowRes",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/graphcast/graphcast_M_m0-m3_tsaf_specloss/"
            ),
        ),
        (
            "SFNO",
            "LowRes",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/sfno/sfno_L_tsaf_specloss/"
            ),
        ),
        # (
        #     "SFNOOld",
        #     Path(
        #         "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker/sfno/sfno_M_oldall_ts_latlon5.625_l10_6h/rollout/scores/ckpt=best_massfixer=scale"
        #     ),
        # ),
        (
            "SwinTransformer",
            "LowRes",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/swintransformer/swintransformer_S_p1w4_tsaf_specloss/"
            ),
        ),
        (
            "SwinTransformer",
            "MidRes",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_midres/swintransformer/swintransformer_S_p1w4_tsaf_specloss_long/"
            ),
        ),
        (
            "SwinTransformer",
            "OrigRes",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_origres/swintransformer/swintransformer_S_p1w4_tsaf_specloss_long/"
            ),
        ),
        (
            "TM5",
            "LowRes",
            Path(
                "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/obs_carbontracker_latlon5.625_l10_6h.zarr"
            ),
        ),
        (
            "TM5",
            "MidRes",
            Path(
                "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/obs_carbontracker_latlon2.8125_l20_6h.zarr"
            ),
        ),
        (
            "TM5",
            "OrigRes",
            Path(
                "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/obs_carbontracker_latlon2x3_l34_3h.zarr"
            ),
        ),
    ]

    global_df, station_df, speed_df = collect_dataframes_from_runs(Models)
    # tm3_df = get_obspack_tm3_dataframe(
    #     "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/obs_carbontracker_latlon5.625_l10_6h.zarr"
    # )
    # %%
    plot_model_intercomparison_barplots(
        global_df,
        station_df,
        speed_df,
        "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/",
        # tm3_df=tm3_df,
        imgformats=["pdf"],
    )
