# %%
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
    global_dfs = []
    for model, path in runs.items():
        try:
            df = pd.read_csv(path / "metrics.csv")
        except:
            continue
        df["model"] = model
        df["years"] = int(model.split("_")[-1][:-1])
        df["Pretraining"] = "LowRes" if model.split("_")[1] == "pretrained" else "None"
        global_dfs.append(df)
    global_df = pd.concat(global_dfs)

    station_dfs = []
    for model, path in runs.items():
        try:
            df = pd.read_csv(path / "obs_metrics.csv")
        except:
            continue
        df["model"] = model
        df["years"] = int(model.split("_")[-1][:-1])
        df["Pretraining"] = "LowRes" if model.split("_")[1] == "pretrained" else "None"
        station_dfs.append(df)
    station_df = pd.concat(station_dfs)
    return global_df, station_df


# %%
def plot_metrics(
    global_df, station_df, out_path, tm3_df=None, imgformats=["svg", "png", "pdf"]
):
    with mpl.rc_context(mpl_rc_params):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), width_ratios=(5, 5))

        ax = axs[0]
        ax2 = ax.twinx()

        blue_color = "#3274a1"
        orange_color = "#e1812b"

        sns.lineplot(
            global_df.rename(
                columns={"Days_R2>0.9_co2molemix": "Decorrelation time [days]"}
            ),
            x="years",
            y="Decorrelation time [days]",
            ax=ax,
            legend=False,
            color=blue_color,
            style="Pretraining",
        )
        sns.lineplot(
            station_df[station_df.obs_filename.str.contains("representative")]
            .rename(columns={"r2": r"$R^2$"})
            .reset_index(),
            x="years",
            y=r"$R^2$",
            ax=ax2,
            legend=False,
            color=orange_color,
            style="Pretraining",
        )

        if tm3_df is not None:
            tm3_r2_mean = tm3_df[
                tm3_df.obs_filename.str.contains("representative")
            ].r2.mean()
            ax2.axhline(tm3_r2_mean, color=orange_color, linestyle=":", zorder=0)
            # ax2.text(
            #     20,
            #     tm3_r2_mean + 0.02,
            #     "TM5",
            #     color="black",
            #     ha="right",
            #     va="center",
            #     zorder=0,
            #     # rotation=90,
            #     fontsize=mpl_rc_params["xtick.labelsize"],
            # )
            ax2.set_yticks(
                [0, 0.2, 0.4, 0.6, tm3_r2_mean, 0.8, 1],
                labels=[0, 0.2, 0.4, 0.6, "TM5", 0.8, 1],
            )

        ax.patch.set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.set_ylabel("Decorrelation time [days]", color=blue_color)
        ax.tick_params(axis="y", labelcolor=blue_color)
        ax2.set_ylabel(r"$R^2$", color=orange_color)
        ax2.tick_params(axis="y", labelcolor=orange_color)
        ax.set_ylim(0, 90)
        ax2.set_ylim(0, 1)
        ax.set_xlabel("Years of MidRes Training Data")
        ax2.set_xlabel("Years of MidRes Training Data")

        station_rmse = station_df[
            station_df.obs_filename.str.contains("representative")
        ][["years", "Pretraining", "rmse"]].rename(columns={"rmse": "RMSE [ppm]"})
        station_rmse["Test Data"] = "Station"
        global_rmse = global_df.rename(
            columns={"RMSE_3D_7d_co2molemix": "RMSE [ppm]"}  # "RMSE_4D_co2molemix"
        )[  # TODO !!!
            ["years", "Pretraining", "RMSE [ppm]"]
        ]
        global_rmse["Test Data"] = "Global"
        rmse_df = pd.concat([station_rmse, global_rmse])

        axs[1].set_ylabel("")
        axs[1].set_yticks([])
        axs[1].set_ylim(0, 5)
        ax3 = axs[1].twinx()
        ax3.set_ylim(0, 10)

        sns.lineplot(
            data=rmse_df,
            x="years",
            y="RMSE [ppm]",
            hue="Test Data",
            hue_order=["Global", "Station"],
            errorbar=("se", 2),
            ax=ax3,
            style="Pretraining",
        )

        if tm3_df is not None:
            tm3_rmse_mean = tm3_df[
                tm3_df.obs_filename.str.contains("representative")
            ].rmse.mean()
            ax3.axhline(tm3_rmse_mean, color=orange_color, linestyle=":", zorder=0)
            # ax3.text(
            #     20,
            #     tm3_rmse_mean - 0.3,
            #     "TM5",
            #     color="black",
            #     ha="right",
            #     va="center",
            #     zorder=0,
            #     # rotation=90,
            #     fontsize=mpl_rc_params["xtick.labelsize"],
            # )
            ax3.set_yticks(
                [0, 1, tm3_rmse_mean, 2.5, 5, 10], labels=[0, 1, "TM5", 2.5, 5, 10]
            )

        axs[1].set_xlabel("Years of MidRes Training Data")
        axs[1].set_xlim(0, 18)
        ax2.set_xlim(0, 18)
        ax3.set_xlabel("Years of MidRes Training Data")
        ax.set_xlim(0, 18)
        ax3.set_xlim(0, 18)

        plt.tight_layout()
        sns.move_legend(
            ax3,
            "upper left",
            bbox_to_anchor=(1.18, 1.0),
            frameon=True,
            facecolor="white",
            framealpha=1,
            # ncols=1,
        )
        for imgformat in imgformats:
            plt.savefig(
                Path(out_path) / f"crossresolution_scores.{imgformat}",
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()


# %%
if __name__ == "__main__":
    Models = {
        f"SFNO_pretrained_{years}y": Path(
            f"/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_midres/sfno/sfno_L_tsaf_specloss_aroma_pretrained_{years}y/rollout/scores/ckpt=best_massfixer=scale"
        )
        for years in [1, 5, 17]  #
    }
    Models |= {
        f"SFNO_scratch_{years}y": Path(
            f"/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_midres/sfno/sfno_L_tsaf_specloss_aroma_{years}y/rollout/scores/ckpt=best_massfixer=scale"
        )
        for years in [1, 5, 17]  #
    }

    global_df, station_df = collect_dataframes_from_runs(Models)

    tm3_df = get_obspack_tm3_dataframe(
        "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/obs_carbontracker_latlon2.8125_l20_6h.zarr"
    )
    plot_metrics(
        global_df,
        station_df,
        "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/",
        tm3_df=tm3_df,
        imgformats=["pdf"],
    )
