"""Comparison plots: Phase 11 vs Phase 24 ensemble eval.

Produces:
  1. error_vs_lead.png : RMSE / CRPS / spread per-lead overlay for
     Phase24-AR, Phase24-SW, Phase11-trajectory.
  2. qualitative_<lead>.png : GT vs 3 ensemble members per model, one figure
     per selected lead.

Inputs it looks for (all xarray-zarr with dims [init, sample, lead, lat, lon, level]):
  - Phase 24 AR         : 24_fm_unet_transport_prior/.../eval_autoregressive/preds_ensemble.zarr
  - Phase 24 SW         : 24_fm_unet_transport_prior/.../eval_slidingwindow_120/preds_ensemble.zarr
  - Phase 11 trajectory : compare_11_24/preds_trajectory/preds_phase11.zarr
  - GT                  : compare_11_24/preds_trajectory/gt.zarr

One-step combined CSV used to annotate the lead=0 marker:
  - compare_11_24/preds_onestep/metrics_per_lead_combined.csv
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print("imports done", flush=True)
import numpy as np
import pandas as pd
import xarray as xr

from neural_transport.inference.analyse import compute_trajectory_ensemble_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
P24 = Path(
    "/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/"
    "data_assimilation/carbontracker_lowres/24_fm_unet_transport_prior/singlestep/preds"
)
ONE = HERE / "preds_onestep"
TRAJ = HERE / "preds_trajectory"
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True, parents=True)

TARGET = "co2massmix"


def load_ds(path):
    if path.exists():
        logger.info("loading %s", path)
        ds = xr.open_zarr(str(path)).load()
        logger.info("  -> loaded %s", dict(ds.sizes))
        return ds
    logger.warning("missing: %s", path)
    return None


def metrics_for(preds, gt):
    per_lead, _, _ = compute_trajectory_ensemble_metrics(gt, preds, target_var=TARGET)
    return per_lead


def error_vs_lead_plot(entries, out_path):
    """entries: list of (label, per_lead_df, color, linestyle)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for label, df, color, ls in entries:
        lead = df.index.values
        axes[0].plot(lead, df["rmse_mean"], color=color, ls=ls, label=label)
        axes[1].plot(lead, df["crps"], color=color, ls=ls, label=label)
        axes[2].plot(lead, df["spread_error_ratio"], color=color, ls=ls, label=label)
    axes[0].set_ylabel("RMSE of ensemble mean")
    axes[1].set_ylabel("CRPS")
    axes[2].set_ylabel("Spread / RMSE")
    for ax in axes:
        ax.set_xlabel("Lead time [steps of 6h]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[2].axhline(1.0, color="k", ls=":", lw=0.8, label="ideal")
    fig.suptitle("Phase 11 vs Phase 24 — error growth over lead time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("saved %s", out_path)


def qualitative_panel(gt, phase11, phase24, lead, out_path, init_slot=0, level_idx=0, n_members=3):
    """GT | 3×Phase11 samples | 3×Phase24 samples at a single lead."""
    gt_map = gt[TARGET].isel(init=init_slot, lead=lead, level=level_idx).values
    p11 = phase11[TARGET].isel(init=init_slot, lead=lead, level=level_idx)
    p24 = phase24[TARGET].isel(init=init_slot, lead=lead, level=level_idx)

    ncols = 1 + 2 * n_members
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3.2))
    vmin, vmax = np.nanpercentile(gt_map, [2, 98])
    axes[0].imshow(gt_map, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"GT (lead={lead})")
    axes[0].axis("off")
    for s in range(n_members):
        axes[1 + s].imshow(p11.isel(sample=s).values, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1 + s].set_title(f"P11 #{s}")
        axes[1 + s].axis("off")
        axes[1 + n_members + s].imshow(p24.isel(sample=s).values, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1 + n_members + s].set_title(f"P24 #{s}")
        axes[1 + n_members + s].axis("off")
    fig.suptitle(f"Qualitative samples @ lead={lead} (init slot {init_slot}, level {level_idx})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("saved %s", out_path)


def main():
    print("main start", flush=True)
    gt_onestep = load_ds(ONE / "gt.zarr")
    p11_one = load_ds(ONE / "preds_phase11.zarr")
    p24_one = load_ds(ONE / "preds_phase24.zarr")

    gt_traj = load_ds(TRAJ / "gt.zarr")
    p11_traj = load_ds(TRAJ / "preds_phase11.zarr")
    p24_traj = load_ds(TRAJ / "preds_phase24.zarr")

    # ── Error-vs-lead overlay ──
    entries = []
    if p11_traj is not None and gt_traj is not None:
        entries.append(("Phase11 (unconditional)", metrics_for(p11_traj, gt_traj), "tab:gray", "-"))
    if p24_traj is not None and gt_traj is not None:
        entries.append(("Phase24 trajectory", metrics_for(p24_traj, gt_traj), "tab:blue", "-"))
    # Phase 24 full-run metrics are precomputed; load from CSV directly.
    for label, csv_path, color in [
        ("Phase24 AR (full)", P24 / "eval_autoregressive" / "scores" / "metrics_per_lead.csv", "tab:orange"),
        ("Phase24 SW-120 (full)", P24 / "eval_slidingwindow_120" / "scores" / "metrics_per_lead.csv", "tab:green"),
    ]:
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col="lead")
            entries.append((label, df, color, "-"))
        else:
            logger.warning("missing %s", csv_path)

    if entries:
        error_vs_lead_plot(entries, OUT / "error_vs_lead.png")

    # ── Qualitative panels (requires trajectory zarrs) ──
    if p11_traj is not None and p24_traj is not None and gt_traj is not None:
        n_mem = min(3, p11_traj.sizes["sample"], p24_traj.sizes["sample"])
        leads_to_show = [0, min(29, gt_traj.sizes["lead"] - 1),
                         min(119, gt_traj.sizes["lead"] - 1),
                         gt_traj.sizes["lead"] - 1]
        for lead in sorted(set(leads_to_show)):
            qualitative_panel(gt_traj, p11_traj, p24_traj, lead,
                              OUT / f"qualitative_lead{lead:04d}.png", n_members=n_mem)

    # ── Save one-step comparison CSV with both models side by side ──
    csv = ONE / "metrics_per_lead_combined.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        print("\nOne-step ensemble comparison:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
