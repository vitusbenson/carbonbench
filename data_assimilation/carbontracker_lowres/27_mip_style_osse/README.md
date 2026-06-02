# P3 — MIP-style OSSE (realistic synthetic XCO₂)

Bridges the idealised OSSE (`25_transport_prior_osse/`) and the real experiment
(P6). Observations are synthesised from a **known CarbonTracker truth** but under
**MIP-realistic observing conditions**:

- **Sampling geometry**: the *real* OCO-2 orbit footprints (which model-grid
  cells were actually observed each 6 h), read from the staged regridded MIP
  product `OCO2MIP_OCO2/train/mip_oco2_latlon5.625_l20_6h.zarr` (covers
  2014-09 → 2020-12 = the MIP period). Realised coverage ≈ **1.5–2.4 %** of grid
  cells per 6 h step — ~15–20× sparser than the idealised OSSE's 30 %.
- **Forward operator**: the corrected P1 *interpolate-then-apply* operator —
  the model CO₂ profile is interpolated (linear in log-p) to the retrieval's
  20 native levels, then the real 20-level averaging kernel + pressure weights
  are applied (`effective_column_kernel` → effective model-grid kernel `g`).
- **Retrieval noise**: optional additive Gaussian noise on the synthetic obs.

The XCO₂ *values* are synthetic (truth → operator → +noise); only the geometry
and the averaging kernels come from real OCO-2.

## Model
Leak-free (P2) residual-FM EnKF:
`25c_v4_residual_fm/phase2p_residual_fm_stable_leakfree` (train ≤2013, val 2014).
The whole MIP test period 2015–2020 is held out of training.

## Run
```bash
# smoke (local GPU, polite)
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 nice -n 10 python eval_mip_osse.py \
    --method enkf --n-inits 4 --n-samples 10 --n-steps 80 --obs-every 4 \
    --enkf-loc-sigma 4 --sigma-obs 0.1 --tag smoke80

# full suite (SLURM): free baseline + orbit-EnKF + realism sweep
sbatch run_mip_osse.slurm
```

## Realism knobs (P3.4 sweep)
- `--ak-mode {real,uniform}` — real 20-level AK shape vs flattened (AK ablation).
- `--thin-fraction f` — keep fraction `f` of real observed cells (sparsity).
- `--obs-noise σ` — additive retrieval-noise std on synthetic obs.
- `--obs-every k` — assimilation cadence (every k×6 h).

## Outputs
`results/<method>_<tag>/`: `preds_trajectory.zarr`, `gt_trajectory.zarr`,
`scores/{metrics_per_lead,metrics_summary,rank_histogram}.csv`, `method_info.json`.

## Key code
- `neural_transport/inference/orbit_obs.py` — `OrbitObsProvider` (real orbit + AK
  reader, keyed by timestamp).
- `neural_transport/inference/generation.py` — `_build_orbit_enkf_obs` + the
  `orbit_obs`/`obs_noise`/`ak_mode`/`thin_fraction` path in
  `generate_trajectory_enkf`.
- `neural_transport/forward_model.py` — `effective_column_kernel` (P1).
