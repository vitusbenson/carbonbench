"""Utility functions for plotting."""

from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs

def save_figure(fig, out_dir, filename, imgformats=["svg", "png", "pdf"], dpi=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    for fmt in imgformats:
        fig.savefig(out_dir / f"{filename}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

mpl_rc_params = {
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
}

def decorate_earth(ax,
                   terrain=False,
                   grid=False,
                   land=False, ocean=False,
                   borders=False, lakes=False, rivers=False):
    """Add optional geographic features to an axis."""
    ax.coastlines()

    if grid:
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="dimgray",
            alpha=0.4, zorder=2
        )
        gl.xlabel_style = {"size": 8, "color": "dimgray"}
        gl.ylabel_style = {"size": 8, "color": "dimgray"}
        gl.bottom_labels = False
        gl.right_labels = False

    if terrain:
        ax.stock_img()
    if land:
        ax.add_feature(cfeature.LAND)
    if ocean:
        ax.add_feature(cfeature.OCEAN)
    if borders:
        ax.add_feature(cfeature.BORDERS, linestyle=':')
    if lakes:
        ax.add_feature(cfeature.LAKES)
    if rivers:
        ax.add_feature(cfeature.RIVERS)

PROJECTION_MAP = {
    "PlateCarree": ccrs.PlateCarree,
    "Robinson": ccrs.Robinson,
    "Mollweide": ccrs.Mollweide,
    "Aitoff": ccrs.Aitoff,
    "InterruptedGoodeHomolosine": ccrs.InterruptedGoodeHomolosine,
    "Orthographic": ccrs.Orthographic,
    "Mercator": ccrs.Mercator,
    "LambertCylindrical": ccrs.LambertCylindrical,
}

def parse_projections(names):
    """Convert list of projection names to ccrs projection instances."""
    if not names:
        return [ccrs.PlateCarree()]
    projections = []
    for name in names:
        if name not in PROJECTION_MAP:
            raise ValueError(f"Unknown projection: {name}. "
                             f"Available: {', '.join(PROJECTION_MAP.keys())}")
        projections.append(PROJECTION_MAP[name]())
    return projections
