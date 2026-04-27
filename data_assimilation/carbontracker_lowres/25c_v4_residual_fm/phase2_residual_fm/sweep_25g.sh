#!/bin/bash
# Phase 25g posterior-conditioning sweep on the residual FM foundation.
# Submits one slurm job per (method, noise_scale) pair. Run after phase 2
# training completes and AR-eval baseline has been recorded.
#
# Usage: bash sweep_25g.sh
#
# Each job writes results into `25_transport_prior_osse/results/<method>_<tag>/`.
set -euo pipefail

EXP_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25_transport_prior_osse

# Unconditional AR baseline on residual FM (no obs)
sbatch --export=METHOD=none,TAG=25g_1month_uncond,PHASE=25g,NSTEPS=120 "$EXP_DIR/run_v2.slurm"
# Unconditional + ρ=1.05 (Phase 25e on the new foundation)
sbatch --export=METHOD=none,TAG=25g_1month_ns105,PHASE=25g,NSTEPS=120,NOISE_SCALE=1.05 "$EXP_DIR/run_v2.slurm"
# Unconditional + ρ=1.10
sbatch --export=METHOD=none,TAG=25g_1month_ns110,PHASE=25g,NSTEPS=120,NOISE_SCALE=1.10 "$EXP_DIR/run_v2.slurm"

# FMPS (per-step DPS)
sbatch --export=METHOD=fmps,TAG=25g_1month_fmps,PHASE=25g,NSTEPS=120,OBS_EVERY=4 "$EXP_DIR/run_v2.slurm"
# FMPS + ρ=1.05
sbatch --export=METHOD=fmps,TAG=25g_1month_fmps_ns105,PHASE=25g,NSTEPS=120,OBS_EVERY=4,NOISE_SCALE=1.05 "$EXP_DIR/run_v2.slurm"

# D-Flow (source-noise opt)
sbatch --export=METHOD=dflow,TAG=25g_1month_dflow,PHASE=25g,NSTEPS=120,OBS_EVERY=4,NOPT=30,CHUNK=50 "$EXP_DIR/run_v2.slurm"
# D-Flow + ρ=1.05  (the prior-best DA combo)
sbatch --export=METHOD=dflow,TAG=25g_1month_dflow_ns105,PHASE=25g,NSTEPS=120,OBS_EVERY=4,NOPT=30,CHUNK=50,NOISE_SCALE=1.05 "$EXP_DIR/run_v2.slurm"
