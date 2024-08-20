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


def plot_3d_variable(
    preds, targs, varname, ts=[4 * 7 - 1, 4 * 30 - 1, 4 * 90 - 1], level=850
):

    unit = targs.attrs.get("units", "")
    long_name = targs.attrs.get("long_name", varname)
    clabel = f"{long_name} [{unit}]"
    clabel_delta = f"Delta [{unit}]"

    pred, targ = get_pred_targ_from_varname(
        preds.isel(time=ts).sel(level=level, method="nearest"),
        targs.isel(time=ts).sel(level=level, method="nearest"),
        varname,
    )

    vmin = targ.quantile(0.02).compute().item()
    vmax = targ.quantile(0.98).compute().item()
    max_delta = 1.0  # np.abs(targ - pred).quantile(0.95).compute().item()

    nstep = 101

    kwargs = dict(
        clabel=clabel,
        clabel_delta=clabel_delta,
        vmin=vmin,
        vmax=vmax,
        max_delta=max_delta,
        nstep=nstep,
    )

    with mpl.rc_context(mpl_rc_params):
        fig, axs = plt.subplots(
            3,
            3,
            # set the height ratios between the rows
            # height_ratios=[1, 2 / 3],
            # width_ratios=[1, 1, 1, 1, 1, 1, 0.25],
            subplot_kw=dict(projection=ccrs.Robinson()),
            gridspec_kw={
                "wspace": 0.05,
                "hspace": 0.0,
                # "bottom": 0.05,
                # "top": 0.95,
                # "left": 0.08,
                # "right": 0.92,
            },
            figsize=(12, 8),
        )

        levels = np.linspace(
            kwargs.get("vmin", targ.min()),
            kwargs.get("vmax", targ.max()),
            kwargs.get("nstep", 21),
        )
        cmap = plt.get_cmap("Spectral_r", len(levels))

        vari_kwargs = dict(
            levels=levels,
            vmin=kwargs.get("vmin", targ.min()),
            vmax=kwargs.get("vmax", targ.max()),
            cmap=cmap,
            zorder=0,
            transform=ccrs.PlateCarree(),
        )

        delta_levels = np.linspace(
            -kwargs.get("max_delta", 1),
            kwargs.get("max_delta", 1),
            kwargs.get("nstep", 21),
        )
        delta_cmap = plt.get_cmap("RdBu_r", len(delta_levels))
        delta_kwargs = dict(
            levels=delta_levels,
            vmin=-kwargs.get("max_delta", 1),
            vmax=kwargs.get("max_delta", 1),
            cmap=delta_cmap,
            zorder=0,
            transform=ccrs.PlateCarree(),
        )

        for i, t in enumerate(ts):
            for j, curr_da in enumerate([targ, pred, targ - pred]):
                # for j, level in enumerate(kwargs.get("levels", [1, 8, 15])):
                ax = axs[i, j]

                cnf = curr_da.isel(time=i).plot(
                    ax=ax,
                    add_colorbar=False,
                    **(vari_kwargs if j < 2 else delta_kwargs),
                )

                ax.set_global()

                gl = ax.gridlines(
                    draw_labels=True,
                    linewidth=0.5,
                    color="dimgray",
                    alpha=0.4,
                    zorder=2,
                )
                gl.xlabel_style = {"size": 6, "color": "dimgray"}
                gl.ylabel_style = {"size": 6, "color": "dimgray"}

                if j < 2:
                    gl.right_labels = False
                gl.geo_labels = False
                if i < 2:
                    gl.bottom_labels = False

                gl.top_labels = False
                gl.left_labels = False

                ax.coastlines(linewidth=0.5, zorder=2)

                ax.set_title(
                    [
                        "Target",
                        "Prediction",
                        "Targ - Pred",
                    ][j]
                    if i == 0
                    else ""
                )
                if j == 0:
                    ax.text(
                        -0.05,
                        0.55,
                        f"{(t+1)//4} days ahead",
                        va="bottom",
                        ha="center",
                        rotation="vertical",
                        rotation_mode="anchor",
                        transform=ax.transAxes,
                        fontsize=mpl_rc_params["ytick.labelsize"],
                    )
                if (i == 2) and (j == 0):
                    fig.tight_layout()
                if (i == 2) and (j > 0):
                    cbar = plt.colorbar(
                        cnf,
                        ax=axs[:, :2] if j < 2 else axs[:, 2:],
                        location="bottom",
                        orientation="horizontal",
                        fraction=0.15,
                        pad=0.05,
                        shrink=0.5 if j < 2 else 1.0,
                        label=f"CO$_2$ molemix [ppm]" if j < 2 else "Delta [ppm]",
                        format=(
                            (lambda x, _: f"{x:.2f}")
                            if j < 2
                            else (lambda x, _: f"{x:.2f}")
                        ),
                        ticks=(
                            np.linspace(delta_kwargs["vmin"], delta_kwargs["vmax"], 5)
                            if j == 2
                            else np.linspace(
                                vari_kwargs["vmin"], vari_kwargs["vmax"], 5
                            )
                        ),
                    )

    return fig


def plot_maps_per_model(
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
    out_dir = Path(out_dir) / "maps_850hpa"
    out_dir.mkdir(exist_ok=True, parents=True)

    model_name = model_name or pred_path.parent.parent.parent.parent.parent.name
    experiment_name = experiment_name or pred_path.parent.parent.parent.parent.name
    singlestep_or_rollout = singlestep_or_rollout or pred_path.parent.parent.parent.name
    ckpt_name = ckpt_name or pred_path.parent.name

    targs, preds = load_pred_targ(target_path, pred_path)

    preds = preds.isel(time=slice(1, None))
    targs = targs.isel(time=slice(1, None)).isel(time=slice(len(preds.time)))

    fig = plot_3d_variable(preds, targs, "co2molemix")
    for imgformat in imgformats:
        plt.savefig(
            out_dir
            / f"{model_name}_{experiment_name}_{singlestep_or_rollout}_{ckpt_name}.{imgformat}",
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

    plot_maps_per_model(
        args.target_path, args.pred_path, args.out_dir, imgformats=["pdf"]
    )
