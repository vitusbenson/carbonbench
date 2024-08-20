from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from neural_transport.inference.plot_results import get_pred_targ_from_varname

mpl_rc_params = {
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
}


def load_preds_targ(target_path, Models):
    targ = xr.open_zarr(target_path)
    co2targ = xr.merge(
        [
            targ["variables_3d"].to_dataset("vari_3d"),
            targ["variables_2d"].to_dataset("vari_2d"),
        ]
    )

    co2preds = {}
    for model, pred_path in Models:
        co2pred = xr.open_zarr(pred_path)
        co2preds[model] = co2pred.isel(time=slice(1, None))

    co2targ = co2targ.isel(time=slice(1, None)).isel(time=slice(len(co2pred.time)))

    return co2targ, co2preds


def plot_rmse(predsdict, targs, varname):

    unit = targs.attrs.get("units", "")
    long_name = targs.attrs.get("long_name", varname)
    clabel = f"{long_name} [{unit}]"
    clabel_delta = f"Delta [{unit}]"

    with mpl.rc_context(mpl_rc_params):
        fig = plt.figure(figsize=(18.03 * 0.8, len(predsdict.keys()) * 6 * 0.8))
        gs = fig.add_gridspec(len(predsdict.keys()), 2, width_ratios=[1.82, 1])

        for i, (model, preds) in enumerate(predsdict.items()):

            ax1 = fig.add_subplot(gs[i, 0], projection=ccrs.Robinson())

            ax1.text(
                -0.1,
                1.0,
                model,
                transform=ax1.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="left",
            )

            pred, targ = get_pred_targ_from_varname(
                preds,
                targs,
                varname,
            )

            se = ((pred - targ) ** 2).compute()

            rmse_map = se.mean(["level", "time"]) ** 0.5
            rmse_latlevel = se.mean(["time", "lon"]) ** 0.5

            vmin = 0  # min(
            #     rmse_map.quantile(0.02).compute().item(),
            #     rmse_latlevel.quantile(0.02).compute().item(),
            # )
            vmax = 2  # max(
            #     rmse_map.quantile(0.98).compute().item(),
            #     rmse_latlevel.quantile(0.98).compute().item(),
            # )

            nstep = 101

            kwargs = dict(
                clabel=clabel,
                clabel_delta=clabel_delta,
                vmin=vmin,
                vmax=vmax,
                nstep=nstep,
            )

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

            ax2 = fig.add_subplot(gs[i, 1])

            old_level = rmse_latlevel.level
            rmse_latlevel["level"] = range(len(old_level))
            old_lat = rmse_latlevel.lat
            rmse_latlevel["lat"] = (
                ccrs.Robinson().transform_points(
                    ccrs.Geodetic(), np.zeros_like(old_lat.values), old_lat.values
                )[:, 1]
                / 1.3523
                / 6.378e6
            )

            rmse_latlevel.plot(
                y="lat",
                x="level",
                ax=ax2,
                add_colorbar=True,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                levels=levels,
                zorder=0,
                cbar_kwargs=dict(label="RMSE [ppm]"),
            )

            tick_lats = np.array([-60, -30, 0, 30, 60])
            tick_vals = (
                ccrs.Robinson().transform_points(
                    ccrs.Geodetic(), np.zeros_like(tick_lats), tick_lats
                )[:, 1]
                / 1.3523
                / 6.378e6
            )
            ax2.set_yticks(
                ticks=tick_vals, labels=["60°S", "30°S", "0°", "30°N", "60°N"]
            )
            ax2.set_xticks(
                ticks=range(len(old_level)), labels=old_level.values.astype("int")
            )
            ax2.set_xlabel(f"Pressure Level [hPa]")
            ax2.set_ylabel("Latitude")
        fig.tight_layout()

    return fig


def plot_rmse_per_model(
    target_path,
    Models,
    out_dir,
    imgformats=["svg", "png", "pdf"],
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    targs, preds = load_preds_targ(target_path, Models)

    fig = plot_rmse(preds, targs, "co2molemix")
    for imgformat in imgformats:
        plt.savefig(
            out_dir / f"rmse_patterns_intercomparison.{imgformat}",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()

    return None


if __name__ == "__main__":

    Models = [
        (
            "UNet",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/unet/unet_M_tsaf_specloss/rollout/preds/ckpt=best_massfixer=scale/co2_pred_rollout_QS.zarr"
            ),
        ),
        (
            "GraphCast",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/graphcast/graphcast_M_m0-m3_tsaf_specloss/rollout/preds/ckpt=best_massfixer=scale/co2_pred_rollout_QS.zarr"  # "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker/graphcast/graphcast_M_targshift_m0-m2_latlon5.625_l10_6h/rollout/preds/ckpt=best_massfixer=scale/co2_pred_rollout_QS.zarr"
            ),
        ),
        (
            "SFNO",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/sfno/sfno_L_tsaf_specloss/rollout/preds/ckpt=best_massfixer=scale/co2_pred_rollout_QS.zarr"
            ),
        ),
        (
            "SwinTransformer",
            Path(
                "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/transport_models/carbontracker_lowres/swintransformer/swintransformer_S_p1w4_tsaf_specloss/rollout/preds/ckpt=best_massfixer=scale/co2_pred_rollout_QS.zarr"
            ),
        ),
    ]

    target_path = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/test/carbontracker_latlon5.625_l10_6h.zarr"
    plot_rmse_per_model(
        target_path,
        Models,
        "/User/homes/vbenson/vbenson/CarbonBench/carbonbench/plotting/first_paper/",
        imgformats=["pdf"],
    )
