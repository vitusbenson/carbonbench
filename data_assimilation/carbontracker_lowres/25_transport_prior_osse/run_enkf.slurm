#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p25i_enkf
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25_transport_prior_osse/slurm_%x_%A.out

# Phase 25i: post-hoc per-cell vertical EnKF on free-running residual-FM ensemble.
# Usage: sbatch --export=SIGMA=0.1,INFL=1.0,OBS_EVERY=1,TAG=25i_enkf_oe1_s01_i10,NSAMPLES=10,SEED=42 run_enkf.slurm

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PYTHONFAULTHANDLER=1

EXP_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25_transport_prior_osse
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker

SIGMA=${SIGMA:-0.1}
INFL=${INFL:-1.0}
LOC=${LOC:-0.0}
PINFL=${PINFL:-1.0}
HYBRID=${HYBRID:-0}
OBS_EVERY=${OBS_EVERY:-1}
NSTEPS=${NSTEPS:-120}
NINITS=${NINITS:-20}
NSAMPLES=${NSAMPLES:-10}
NOISE_SCALE=${NOISE_SCALE:-1.05}
PHASE=${PHASE:-25g}
SEED=${SEED:-42}
CHUNK=${CHUNK:-200}
OBS_FRACTION=${OBS_FRACTION:-}
TAG=${TAG:-25i_enkf_oe${OBS_EVERY}_s${SIGMA}_i${INFL}}

EXTRA="--seed ${SEED}"
if [ -n "${OBS_FRACTION}" ]; then EXTRA="${EXTRA} --obs-fraction ${OBS_FRACTION}"; fi
if [ "${HYBRID}" = "1" ]; then EXTRA="${EXTRA} --enkf-hybrid"; fi
if [ "${BIAS}" = "1" ]; then EXTRA="${EXTRA} --enkf-global-bias-correct"; fi
if [ "${EMA}" = "1" ]; then EXTRA="${EXTRA} --ema"; fi

cd "${EXP_DIR}"
srun "${PYBIN}" -u eval_trajectory_v2.py \
    --method enkf \
    --data-root "${DATA_ROOT}" \
    --n-inits "${NINITS}" --n-samples "${NSAMPLES}" --n-steps "${NSTEPS}" \
    --obs-every "${OBS_EVERY}" \
    --sigma-obs "${SIGMA}" --enkf-inflation "${INFL}" --enkf-loc-sigma "${LOC}" --enkf-prior-inflation "${PINFL}" \
    --chunk-size "${CHUNK}" \
    --tag "${TAG}" \
    --phase "${PHASE}" \
    --noise-scale "${NOISE_SCALE}" \
    ${EXTRA}
