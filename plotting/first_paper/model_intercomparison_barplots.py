from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
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
    for model, path in runs:
        df = pd.read_csv(path / "metrics.csv")
        df["model"] = model
        try:
            df2 = pd.read_csv(path / "metrics_None.csv")
            df["Days_R2>0.9_co2molemix"] = df2["Days_R2>0.9_co2molemix"]
        except:
            print("No metrics_None.csv found")
        global_dfs.append(df)
    global_df = pd.concat(global_dfs)
    global_df["model"] = pd.Categorical(
        global_df["model"],
        categories=sorted(sort_order, key=sort_order.get),
        ordered=True,
    )

    station_dfs = []
    for model, path in runs:
        df = pd.read_csv(path / "obs_metrics.csv")
        df["model"] = model
        station_dfs.append(df)
    station_df = pd.concat(station_dfs)
    station_df["model"] = pd.Categorical(
        station_df["model"],
        categories=sorted(sort_order, key=sort_order.get),
        ordered=True,
    )
    return global_df, station_df


def plot_model_intercomparison_barplots(
    global_df,
    station_df,
    out_path,
    tm3_df=None,
    imgformats=["svg", "png", "pdf"],
):
    with mpl.rc_context(mpl_rc_params):
        fig, axs = plt.subplots(1, 2, figsize=(10, 6), width_ratios=(5, 5))

        ax = axs[0]
        ax2 = ax.twinx()

        blue_color = "#3274a1"
        orange_color = "#e1812b"

        global_df.rename(
            columns={"Days_R2>0.9_co2molemix": "Decorrelation time [days]"}
        ).plot(
            kind="bar",
            x="model",
            y="Decorrelation time [days]",
            ax=ax,
            width=0.4,
            position=1,
            legend=False,
            color=blue_color,
            capsize=3,
        )
        station_groupby = (
            station_df[station_df.obs_filename.str.contains("representative")][
                ["model", "r2"]
            ]
        ).groupby("model")

        n_models = len(station_groupby)

        means = station_groupby.mean().reset_index()
        stds = (
            2
            * (
                station_groupby.std() / np.sqrt(station_groupby.count())
            ).values.flatten()
        )

        means.rename(columns={"r2": r"$R^2$"}).plot(
            kind="bar",
            x="model",
            y=r"$R^2$",
            ax=ax2,
            width=0.4,
            position=0,
            legend=False,
            yerr=stds,
            color=orange_color,
            capsize=3,
            error_kw=dict(ecolor="#424242", lw=2.5, capsize=4, capthick=2.5),
        )

        if tm3_df is not None:
            tm3_r2_mean = tm3_df[
                tm3_df.obs_filename.str.contains("representative")
            ].r2.mean()
            tm3_r2_se = tm3_df[
                tm3_df.obs_filename.str.contains("representative")
            ].r2.std() / np.sqrt(
                tm3_df[tm3_df.obs_filename.str.contains("representative")].r2.count()
            )
            ax2.axhspan(
                ymin=tm3_r2_mean - 2 * tm3_r2_se,
                ymax=tm3_r2_mean + 2 * tm3_r2_se,
                color=orange_color,
                alpha=0.3,
                zorder=-3,
            )
            ax2.axhline(tm3_r2_mean, color="black", linestyle="--", zorder=-3)
            ax2.set_yticks(
                [0, 0.2, 0.4, 0.6, tm3_r2_mean, 0.8, 1],
                labels=[0, 0.2, 0.4, 0.6, "TM5", 0.8, 1],
            )
            # ax2.text(
            #     0.7,
            #     tm3_r2_mean + 0.7 * tm3_r2_se,
            #     "TM3",
            #     color="black",
            #     ha="center",
            #     va="center",
            #     zorder=-3,
            #     # rotation=90,
            #     fontsize=mpl_rc_params["xtick.labelsize"],
            # )
        ax.patch.set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.set_ylabel("Decorrelation time [days]", color=blue_color)
        ax.tick_params(axis="y", labelcolor=blue_color)
        ax2.set_ylabel(r"$R^2$", color=orange_color)
        ax2.tick_params(axis="y", labelcolor=orange_color)
        ax.set_ylim(0, 1000)
        ax2.set_ylim(0, 1)
        ax.set_xlabel("")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="center")
        ax.set_xlim(-0.5, n_models - 0.5)
        # plt.xticks(rotation=20, ha="center")

        station_rmse = (
            station_df[station_df.obs_filename.str.contains("representative")][
                ["model", "rmse"]
            ]
            # .groupby("model")
            # .mean()
            # .reset_index()
            .rename(columns={"rmse": "RMSE [ppm]"})
        )
        station_rmse["Test Data"] = "Station"
        global_rmse = global_df.rename(columns={"RMSE_4D_co2molemix": "RMSE [ppm]"})[
            ["model", "RMSE [ppm]"]
        ]
        global_rmse["Test Data"] = "Global"
        rmse_df = pd.concat([station_rmse, global_rmse])

        axs[1].set_ylabel("")
        axs[1].set_yticks([])
        ax3 = axs[1].twinx()

        sns.barplot(
            data=rmse_df,
            x="model",
            y="RMSE [ppm]",
            hue="Test Data",
            hue_order=["Global", "Station"],
            errorbar=("se", 2),
            ax=ax3,
            capsize=0.1,
        )

        if tm3_df is not None:
            tm3_rmse_mean = tm3_df[
                tm3_df.obs_filename.str.contains("representative")
            ].rmse.mean()
            tm3_rmse_se = tm3_df[
                tm3_df.obs_filename.str.contains("representative")
            ].rmse.std() / np.sqrt(
                tm3_df[tm3_df.obs_filename.str.contains("representative")].rmse.count()
            )
            ax3.axhspan(
                ymin=tm3_rmse_mean - 2 * tm3_rmse_se,
                ymax=tm3_rmse_mean + 2 * tm3_rmse_se,
                color=orange_color,
                alpha=0.3,
                zorder=0,
            )
            ax3.axhline(tm3_rmse_mean, color="black", linestyle="--", zorder=0)
            ax3.set_yticks(
                [0, 1, 2, tm3_rmse_mean, 3, 4], labels=[0, 1, 2, "TM5", 3, 4]
            )
            # ax3.text(
            #     -0.3,
            #     tm3_rmse_mean + 0.5 * tm3_rmse_se,
            #     "TM3",
            #     color="black",
            #     ha="center",
            #     va="center",
            #     zorder=0,
            #     # rotation=90,
            #     fontsize=mpl_rc_params["xtick.labelsize"],
            # )

        ax3.set_xlim(-0.5, n_models - 0.5)
        ax3.set_xlabel("")
        ax3.set_ylim(0, 4)
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=20, ha="center")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=20, ha="center")
        plt.tight_layout()
        sns.move_legend(
            ax3,
            "upper center",
            bbox_to_anchor=(0.05, 1),
            frameon=True,
            facecolor="white",
            framealpha=1,
        )
        for imgformat in imgformats:
            plt.savefig(
                Path(out_path) / f"model_intercomparison_barplots.{imgformat}",
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()


if __name__ == "__main__":
    Models = [
        # (
        #     "UNet",
        #     Path(
        #         "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker/unet/unet_M_targshift_latlon5.625_l10_6h/rollout/scores/ckpt=best_massfixer=scale"
        #     ),
        # ),
        (
            "UNet",  # "UNetAroma",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/unet/unet_M_tsaf_specloss/rollout/scores/ckpt=best_massfixer=scale"
            ),
        ),
        (
            "GraphCast",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/graphcast/graphcast_M_m0-m3_tsaf_specloss/rollout/scores/ckpt=best_massfixer=scale"  # "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker/graphcast/graphcast_M_targshift_m0-m2_latlon5.625_l10_6h/rollout/scores/ckpt=best_massfixer=scale"
            ),
        ),
        (
            "SFNO",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/sfno/sfno_L_tsaf_specloss/rollout/scores/ckpt=best_massfixer=scale"
            ),
        ),
        # (
        #     "SFNOHR",
        #     Path(
        #         "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_midres/sfno/sfno_L_tsaf_specloss_long/rollout/scores/ckpt=best_massfixer=scale"
        #     ),
        # ),
        (
            "SwinTransformer",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/swintransformer/swintransformer_S_p1w4_tsaf_specloss/rollout/scores/ckpt=best_massfixer=scale"
            ),
        ),
    ]

    global_df, station_df = collect_dataframes_from_runs(Models)
    tm3_df = get_obspack_tm3_dataframe(
        "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/obs_carbontracker_latlon5.625_l10_6h.zarr"  # "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/obs_carbontracker_latlon2x3_l34_3h.zarr"  #
    )
    plot_model_intercomparison_barplots(
        global_df,
        station_df,
        "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/",
        tm3_df=tm3_df,
        imgformats=["pdf"],
    )
