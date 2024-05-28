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
        df = pd.read_csv(path / "metrics.csv")
        df["model"] = model
        df["years"] = int(model.split("_")[-1][:-1])
        global_dfs.append(df)
    global_df = pd.concat(global_dfs)

    station_dfs = []
    for model, path in runs.items():
        df = pd.read_csv(path / "obs_metrics.csv")
        df["model"] = model
        df["years"] = int(model.split("_")[-1][:-1])
        station_dfs.append(df)
    station_df = pd.concat(station_dfs)
    return global_df, station_df


# %%
def plot_metrics(
    global_df, station_df, out_path, tm3_df=None, imgformats=["svg", "png", "pdf"]
):
    with mpl.rc_context(mpl_rc_params):
        fig, axs = plt.subplots(1, 2, figsize=(10, 6), width_ratios=(5, 5))

        ax = axs[0]
        ax2 = ax.twinx()

        blue_color = "#3274a1"
        orange_color = "#e1812b"

        sns.lineplot(
            global_df.rename(
                columns={"Days_R2>0.8_co2molemix": "Decorrelation time [days]"}
            ),
            x="years",
            y="Decorrelation time [days]",
            ax=ax,
            legend=False,
            color=blue_color,
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
        )

        if tm3_df is not None:
            tm3_r2_mean = tm3_df[
                tm3_df.obs_filename.str.contains("representative")
            ].r2.mean()
            ax2.axhline(tm3_r2_mean, color=orange_color, linestyle="--", zorder=0)
            ax2.text(
                -0.5,
                tm3_r2_mean + 0.02,
                "TM3",
                color="black",
                ha="left",
                va="center",
                zorder=0,
                # rotation=90,
                fontsize=mpl_rc_params["xtick.labelsize"],
            )

        ax.patch.set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.set_ylabel("Decorrelation time [days]", color=blue_color)
        ax.tick_params(axis="y", labelcolor=blue_color)
        ax2.set_ylabel(r"$R^2$", color=orange_color)
        ax2.tick_params(axis="y", labelcolor=orange_color)
        ax.set_ylim(0, 90)
        ax2.set_ylim(0, 1)
        ax.set_xlabel("Years of Training Data")
        ax2.set_xlabel("Years of Training Data")

        station_rmse = station_df[
            station_df.obs_filename.str.contains("representative")
        ][["years", "rmse"]].rename(columns={"rmse": "RMSE [ppm]"})
        station_rmse["Test Data"] = "Station"
        global_rmse = global_df.rename(columns={"RMSE_4D_co2molemix": "RMSE [ppm]"})[
            ["years", "RMSE [ppm]"]
        ]
        global_rmse["Test Data"] = "Global"
        rmse_df = pd.concat([station_rmse, global_rmse])

        axs[1].set_ylabel("")
        axs[1].set_yticks([])
        ax3 = axs[1].twinx()

        sns.lineplot(
            data=rmse_df,
            x="years",
            y="RMSE [ppm]",
            hue="Test Data",
            hue_order=["Global", "Station"],
            errorbar=("se", 2),
            ax=ax3,
        )

        if tm3_df is not None:
            tm3_rmse_mean = tm3_df[
                tm3_df.obs_filename.str.contains("representative")
            ].rmse.mean()
            ax3.axhline(tm3_rmse_mean, color=orange_color, linestyle="--", zorder=0)
            ax3.text(
                -0.5,
                tm3_rmse_mean - 0.3,
                "TM3",
                color="black",
                ha="left",
                va="center",
                zorder=0,
                # rotation=90,
                fontsize=mpl_rc_params["xtick.labelsize"],
            )

        axs[1].set_xlabel("Years of Training Data")
        ax3.set_xlabel("Years of Training Data")

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
                Path(out_path) / f"years_of_train_data.{imgformat}",
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()


# %%
if __name__ == "__main__":
    Models = {
        f"SFNO_{years}y": Path(
            f"/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240517/sfno_L_{years}y"
        )
        for years in [1, 2, 5, 10, 25]  #
    }
    Models["SFNO_39y"] = Path(
        "/User/homes/vbenson/vbenson/graph_tm/experiments/carboscope/transport/runs_20240309/sfno_L_addflux"
    )

    global_df, station_df = collect_dataframes_from_runs(Models)

    tm3_df = get_obspack_tm3_dataframe(
        "/User/homes/vbenson/vbenson/graph_tm/data/Carboscope/test/obspack_carboscope.zarr"
    )
    plot_metrics(
        global_df,
        station_df,
        "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/how_many_training_years/",
        tm3_df=tm3_df,
    )
