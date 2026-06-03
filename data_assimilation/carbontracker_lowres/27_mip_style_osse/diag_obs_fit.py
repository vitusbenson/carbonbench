"""Isolation test: does the conditioned ensemble MEAN reproduce the obs?

A valid posterior mean must fit the observations at observed cells (to ~sigma).
This bypasses the AR rollout: take one state, synthesise orbit obs from the
truth via g, run ONE conditioned generation (FlowDPS / FMPS), and compare the
conditioned ensemble-mean column g.x to the obs at observed cells — vs the free
(unconditioned) ensemble mean. If conditioned is not closer to obs than free,
the obs application is broken (not a fundamental limit).
"""

import argparse
import numpy as np
import torch

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader
from neural_transport.inference.generation import _build_orbit_enkf_obs
from neural_transport.inference.orbit_obs import OrbitObsProvider
from neural_transport.training.train import load_model

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "25_transport_prior_osse"))
from configs import adapt_for_trajectory, load_method_config  # noqa: E402

MODEL_DIR = "/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2p_residual_fm_stable_leakfree"
DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker_leakfree"
ORBIT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/OCO2MIP_OCO2/train/mip_oco2_latlon5.625_l20_6h.zarr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", default="flowdps")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--init", type=int, default=2304)
    ap.add_argument("--sigma-obs", type=float, default=0.05)
    ap.add_argument("--spatial-smoothing", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = load_model(MODEL_DIR, ckpt="best", device=args.device)
    inner = getattr(model, "model", model)
    dc = DataConfig(dataset="carbontracker", grid="latlon5.625", vertical_levels="l10", freq="6h",
                    target_vars=["co2massmix", "p_bottom", "p_top"], forcing_vars=["co2massmix", "u", "v"])
    loader = InferenceDataLoader(dc, data_path=f"{DATA_ROOT}/test"); loader.load_dataset()
    nlat, nlon = loader.grid_info.nlat, loader.grid_info.nlon
    N, C = nlat * nlon, len(loader.grid_info.levels)
    times = loader.dataset.ds.time.values
    prov = OrbitObsProvider(ORBIT, nlat=nlat, nlon=nlon)

    k = args.init
    cur = loader.get_batch(k, device=args.device)          # state x_t  [1,1,N,C]
    nxt = loader.get_batch(k + 1, device=args.device)       # truth x_{t+1}
    gt = nxt["co2massmix"].view(1, N, C)
    pb = nxt["p_bottom"].view(1, N, C); pt = nxt["p_top"].view(1, N, C)
    ts = times[k + 1]
    rng = torch.Generator(device=args.device).manual_seed(0)
    mask, yobs, g = _build_orbit_enkf_obs(prov, [ts], gt, pb, pt, nlat, nlon, device=args.device, rng=rng)
    m = mask[0]  # [N]
    nobs = int(m.sum())
    if nobs == 0:
        print("no obs at this init; pick another --init"); return

    # Build batch (n_samples copies of x_t) + obs conditioning fields.
    batch = {kk: (vv.expand(args.n_samples, *vv.shape[1:]).clone() if torch.is_tensor(vv) else vv)
             for kk, vv in cur.items()}
    batch["co2massmix"] = batch["co2massmix"]
    g_b = g.repeat_interleave(args.n_samples, 0)  # [ns,N,C]
    batch["obs_mask"] = m.view(1, 1, N, 1).expand(args.n_samples, 1, N, 1).clone()
    ov = torch.nan_to_num(yobs[0], nan=0.0).view(1, 1, N, 1).expand(args.n_samples, 1, N, 1).clone()
    batch["xco2_averaging_kernel"] = g_b.view(args.n_samples, 1, N, C)
    batch["pressure_weight"] = torch.ones(args.n_samples, 1, N, C, device=args.device)
    batch["xco2_apriori"] = torch.zeros(args.n_samples, 1, N, 1, device=args.device)
    batch["co2_profile_apriori"] = torch.zeros(args.n_samples, 1, N, C, device=args.device)
    ovn = inner.normalize_observations(ov, batch, target_var="co2massmix", targshift=False)
    batch["obs_values"] = torch.nan_to_num(ovn, nan=0.0)

    cfg = load_method_config(args.sampler, n_samples=args.n_samples)
    sk = adapt_for_trajectory(cfg, n_samples=args.n_samples, method=args.sampler)
    sk["sigma_obs"] = args.sigma_obs
    sk["spatial_smoothing_sigma"] = args.spatial_smoothing

    def colmean(field):  # field [ns,N,C] -> ensemble-mean column g.x at obs cells (physical)
        gm = (g_b * field).sum(-1)  # [ns,N]
        return gm.mean(0)[m]        # [nobs]

    yo = yobs[0][m]  # obs (physical)

    # Free (unconditioned) ensemble.
    inner.generating = True
    inner.generate_kwargs = {"n_samples": 1, "masking": False, "noise_scale": 1.05}
    with torch.no_grad():
        free = model(batch, mode="generate")["co2massmix"].detach().view(args.n_samples, N, C)
    # Conditioned ensemble.
    inner.generate_kwargs = sk
    with torch.no_grad():
        cond = model(batch, mode="generate")["co2massmix"].detach().view(args.n_samples, N, C)

    yf, yc = colmean(free), colmean(cond)
    rmse = lambda a, b: float(torch.sqrt(((a - b) ** 2).mean()))
    print(f"sampler={args.sampler} init={k} nobs={nobs} sigma_obs={args.sigma_obs}")
    print(f"  obs-fit RMSE (g.mean vs obs) @ obs cells:")
    print(f"    FREE (no DA)      : {rmse(yf, yo):.4f}")
    print(f"    CONDITIONED       : {rmse(yc, yo):.4f}   (should be << free if obs applied right)")
    print(f"  obs spread: {float(yo.std()):.3f}; free col spread: {float(yf.std()):.3f}")
    print(f"  mean(obs)={float(yo.mean()):.3f}  mean(free col)={float(yf.mean()):.3f}  mean(cond col)={float(yc.mean()):.3f}")


if __name__ == "__main__":
    main()
