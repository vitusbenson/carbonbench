#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p25b_da
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25_transport_prior_osse/slurm_%x_%A_%a.out

# Usage:  sbatch --export=METHOD=fmps,TAG=1month_v2,NSTEPS=120,NOPT=30 run_v2.slurm
#
# METHOD: none | fmps | dflow
# TAG:    1month_v2 | full_v2
# NSTEPS: 120 (1 month) or 1460 (full year)
# NOPT:   D-Flow n_opt_steps (default 30 for trajectory)

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PYTHONFAULTHANDLER=1

EXP_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25_transport_prior_osse
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker

METHOD=${METHOD:-fmps}
TAG=${TAG:-1month_v2}
NSTEPS=${NSTEPS:-120}
NINITS=${NINITS:-20}
NSAMPLES=${NSAMPLES:-10}
OBS_EVERY=${OBS_EVERY:-4}
NOPT=${NOPT:-30}
CHUNK=${CHUNK:-100}
PHASE=${PHASE:-24}
NOISE_SCALE=${NOISE_SCALE:-1.0}

EXTRA=""
if [ "${METHOD}" = "dflow" ]; then EXTRA="--n-opt-steps ${NOPT}"; fi

cd "${EXP_DIR}"
srun "${PYBIN}" -u eval_trajectory_v2.py \
    --method "${METHOD}" \
    --data-root "${DATA_ROOT}" \
    --n-inits "${NINITS}" \
    --n-samples "${NSAMPLES}" \
    --n-steps "${NSTEPS}" \
    --obs-every "${OBS_EVERY}" \
    --chunk-size "${CHUNK}" \
    --tag "${TAG}" \
    --phase "${PHASE}" \
    --noise-scale "${NOISE_SCALE}" \
    ${EXTRA}
