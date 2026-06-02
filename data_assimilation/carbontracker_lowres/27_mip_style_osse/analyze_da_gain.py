"""Diagnose where (and whether) DA actually helps in the MIP-style OSSE.

A few-% global-field gain is suspicious. This decomposes the free-vs-EnKF
RMSE to localise the gain:
  (1) global 3-D field   vs   (2) at observed columns only
  (3) full 3-D field     vs   (4) XCO2 column space
and reports cumulative observed coverage + the per-lead analysis-step gain at
observed cells. It also frames the identical-twin issue (model trained on the
CT truth → small forecast error to correct).

Uses the saved preds/gt zarrs (ensemble mean) + reconstructs the real orbit
masks from the run's init_indices via OrbitObsProvider.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.orbit_obs import OrbitObsProvider

EXP_DIR = Path(__file__).resolve().parent
DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker_leakfree"
ORBIT_ZARR = ("/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/OCO2MIP_OCO2/train/"
              "mip_oco2_latlon5.625_l20_6h.zarr")

# Nominal l10 layer pressure weights h_l = dp/psurf (surface->top), from the
# CarbonTracker grid; used to project the 3-D field to an XCO2 column.
P_BOT = np.array([966.8, 959.3, 949.5, 926.8, 900.8, 806.9, 619.1, 430.3, 240.4, 73.1])
P_TOP = np.array([959.3, 949.5, 926.8, 900.8, 806.9, 619.1, 430.3, 240.4, 73.1, 0.0])
H_L = (P_BOT - P_TOP) / P_BOT[0]  # sums to ~1


def wrmse(err, w, mask=None):
    """Area-weighted RMS of err[...lat,lon(,level)] using lat weights w[lat].
    If mask[lat,lon] given (per init,lead), restrict to True cells."""
    # err: [I, L, lat, lon, lev]; w: [lat]
    wgrid = np.broadcast_to(w[None, None, :, None, None], err.shape)
    if mask is not None:
        m = np.broadcast_to(mask[..., None], err.shape)  # [I,L,lat,lon,lev]
        wgrid = wgrid * m
    num = np.nansum(wgrid * err**2)
    den = np.nansum(wgrid)
    return float(np.sqrt(num / den)) if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", default="none_full")
    ap.add_argument("--enkf", default="enkf_full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--obs-every", type=int, default=4)
    args = ap.parse_args()

    res = EXP_DIR / "results"
    info = json.loads((res / args.enkf / "method_info.json").read_text())
    init_indices = info["init_indices"]
    n_steps = info["n_steps"]

    free = xr.open_zarr(res / args.free / "preds_trajectory.zarr")["co2massmix"].mean("sample").values
    enkf = xr.open_zarr(res / args.enkf / "preds_trajectory.zarr")["co2massmix"].mean("sample").values
    gt = xr.open_zarr(res / args.enkf / "gt_trajectory.zarr")["co2massmix"].values
    # shapes [I, L, lat, lon, lev]
    I, L, nlat, nlon, nlev = gt.shape

    dc = DataConfig(dataset="carbontracker", grid="latlon5.625", vertical_levels="l10",
                    freq="6h", target_vars=["co2massmix", "p_bottom", "p_top"],
                    forcing_vars=["co2massmix", "u", "v"])
    loader = InferenceDataLoader(dc, data_path=f"{DATA_ROOT}/{args.split}")
    loader.load_dataset()
    times = loader.dataset.ds.time.values
    w = np.cos(np.radians(loader.grid_info.lat))

    prov = OrbitObsProvider(ORBIT_ZARR, nlat=nlat, nlon=nlon)
    # obs mask per (init, lead): observed at the *analysis* time = times[init+lead+1].
    obs_mask = np.zeros((I, L, nlat, nlon), bool)
    obs_present = np.array([(k % args.obs_every == 0) for k in range(L)])
    for i, init in enumerate(init_indices):
        for k in range(L):
            if not obs_present[k]:
                continue
            j = init + k + 1
            if j < len(times):
                rec = prov.get(times[j])
                if rec is not None:
                    obs_mask[i, k] = rec["mask"]

    # cumulative-ever-observed mask per init (any lead up to k).
    ever = np.cumsum(obs_mask.astype(int), axis=1) > 0  # [I, L, lat, lon]

    ef, ee = gt - free, gt - enkf  # errors
    # XCO2 column projections.
    xf = (free * H_L).sum(-1); xe = (enkf * H_L).sum(-1); xg = (gt * H_L).sum(-1)
    exf = (xg - xf)[..., None]; exe = (xg - xe)[..., None]  # [I,L,lat,lon,1]

    def g(a, b):
        return f"{a:.4f} -> {b:.4f}  ({100*(a-b)/a:+.1f}%)"

    obs_only = obs_mask.copy()
    obs_leads = obs_present

    print(f"\n=== DA-gain decomposition ({args.free} vs {args.enkf}) ===")
    print(f"inits={I} leads={L} grid={nlat}x{nlon}x{nlev}  obs_every={args.obs_every}")
    cov_step = obs_mask[:, obs_leads].mean()
    cov_end = ever[:, -1].mean()
    print(f"coverage: {100*cov_step:.2f}% of cells per obs-step; "
          f"{100*cov_end:.1f}% of cells observed AT LEAST ONCE by lead {L-1}")

    print("\n3-D field RMSE (all leads):")
    print("  global              ", g(wrmse(ef, w), wrmse(ee, w)))
    print("  observed cells only ", g(wrmse(ef, w, obs_only.reshape(I, L, nlat, nlon)),
                                       wrmse(ee, w, obs_only.reshape(I, L, nlat, nlon))))
    print("  ever-observed cells ", g(wrmse(ef, w, ever), wrmse(ee, w, ever)))

    print("\nXCO2 column RMSE (all leads):")
    print("  global              ", g(wrmse(exf, w), wrmse(exe, w)))
    print("  observed cells only ", g(wrmse(exf, w, obs_only), wrmse(exe, w, obs_only)))

    # At obs leads, error AT observed cells (analysis quality) vs elsewhere.
    of = ef[:, obs_leads]; oe = ee[:, obs_leads]; om = obs_mask[:, obs_leads]
    oxf = exf[:, obs_leads]; oxe = exe[:, obs_leads]
    print("\nAt obs-steps only:")
    print("  3-D @ observed cells", g(wrmse(of, w, om), wrmse(oe, w, om)))
    print("  XCO2 @ observed cells", g(wrmse(oxf, w, om), wrmse(oxe, w, om)))
    unobs = ~obs_mask[:, obs_leads]
    print("  3-D @ UNobserved    ", g(wrmse(of, w, unobs), wrmse(oe, w, unobs)))

    # Identical-twin framing: the free run's error IS the model's CT-emulation
    # error (truth=CT, model trained on CT) — small, so little headroom for DA.
    print("\nIdentical-twin framing — free XCO2 RMSE vs lead (model emulating CT):")
    for k in [0, 4, 20, 40, 80, L - 1]:
        e = wrmse((xg - xf)[:, k:k + 1][..., None], w)
        print(f"  lead {k:3d}: {e:.3f} ppm")
    print("  (free error stays ~sub-ppm-to-1ppm in column space; truth=CT and the\n"
          "   model is trained on CT, so the forecast is already near-perfect ->\n"
          "   structurally small global DA headroom. Gain concentrates at obs cells.)")


if __name__ == "__main__":
    main()
