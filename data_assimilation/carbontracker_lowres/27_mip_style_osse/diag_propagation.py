"""Does the DA signal propagate from observed cells into the global field?

Tests the hypothesis that transport should carry local corrections into the
rest of the field over time. Computes, per lead, the free-vs-EnKF RMSE gain at
cells that have NEVER been observed up to that lead (pure propagation), vs at
ever-observed cells. If the never-observed gain grows with lead, transport is
propagating; if flat, the analysis isn't translating to the global field.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.orbit_obs import OrbitObsProvider

EXP = Path(__file__).resolve().parent
DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker_leakfree"
ORBIT = ("/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/OCO2MIP_OCO2/train/"
         "mip_oco2_latlon5.625_l20_6h.zarr")
P_BOT = np.array([966.8, 959.3, 949.5, 926.8, 900.8, 806.9, 619.1, 430.3, 240.4, 73.1])
P_TOP = np.array([959.3, 949.5, 926.8, 900.8, 806.9, 619.1, 430.3, 240.4, 73.1, 0.0])
H_L = (P_BOT - P_TOP) / P_BOT[0]


def wrmse_lead(err, w, mask):  # err [I,lat,lon], mask [I,lat,lon]
    wg = np.broadcast_to(w[None, :, None], err.shape) * mask
    num = np.nansum(wg * err**2); den = np.nansum(wg)
    return float(np.sqrt(num / den)) if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", default="none_full")
    ap.add_argument("--enkf", default="enkf_full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--obs-every", type=int, default=4)
    args = ap.parse_args()
    res = EXP / "results"
    info = json.loads((res / args.enkf / "method_info.json").read_text())
    inits = info["init_indices"]

    free = xr.open_zarr(res / args.free / "preds_trajectory.zarr")["co2massmix"].mean("sample").values
    enkf = xr.open_zarr(res / args.enkf / "preds_trajectory.zarr")["co2massmix"].mean("sample").values
    gt = xr.open_zarr(res / args.enkf / "gt_trajectory.zarr")["co2massmix"].values
    I, L, nlat, nlon, _ = gt.shape

    dc = DataConfig(dataset="carbontracker", grid="latlon5.625", vertical_levels="l10", freq="6h",
                    target_vars=["co2massmix", "p_bottom", "p_top"], forcing_vars=["co2massmix", "u", "v"])
    loader = InferenceDataLoader(dc, data_path=f"{DATA_ROOT}/{args.split}"); loader.load_dataset()
    times = loader.dataset.ds.time.values
    w = np.cos(np.radians(loader.grid_info.lat))
    prov = OrbitObsProvider(ORBIT, nlat=nlat, nlon=nlon)

    obs_mask = np.zeros((I, L, nlat, nlon), bool)
    for i, init in enumerate(inits):
        for k in range(L):
            if k % args.obs_every == 0 and init + k + 1 < len(times):
                rec = prov.get(times[init + k + 1])
                if rec is not None:
                    obs_mask[i, k] = rec["mask"]
    ever = np.cumsum(obs_mask, axis=1) > 0  # observed at or before lead k

    # XCO2 column error per lead.
    xf = (free * H_L).sum(-1); xe = (enkf * H_L).sum(-1); xg = (gt * H_L).sum(-1)

    print(f"inits={I} leads={L}  obs_every={args.obs_every}")
    print(f"{'lead':>4} {'%seen':>6} | {'never-obs free->enkf (gain)':>30} | {'ever-obs gain':>14}")
    errf = xg - xf  # [I,L,lat,lon] free XCO2 error
    erre = xg - xe  # EnKF XCO2 error
    for k in [4, 12, 24, 48, 72, 96, 119]:
        never = ~ever[:, k]            # [I,lat,lon] cells never seen up to lead k
        seen = ever[:, k]
        nf = wrmse_lead(errf[:, k], w, never); ne = wrmse_lead(erre[:, k], w, never)
        sf = wrmse_lead(errf[:, k], w, seen); se = wrmse_lead(erre[:, k], w, seen)
        pct = 100 * seen.mean()
        ng = 100 * (nf - ne) / nf if nf else 0
        sg = 100 * (sf - se) / sf if sf else 0
        print(f"{k:>4} {pct:6.1f} | {nf:6.3f} -> {ne:6.3f}  ({ng:+5.1f}%) | ({sg:+5.1f}%)")


if __name__ == "__main__":
    main()
