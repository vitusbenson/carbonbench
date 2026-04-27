#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p25g_p2_eval_ar
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2_residual_fm/slurm_%x_%j.out

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PATH=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin:$PATH
EXP_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2_residual_fm
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker

NOISE_SCALE=${NOISE_SCALE:-1.0}
TAG=${TAG:-residual_fm_1month}

cd "${EXP_DIR}"
srun "${PYBIN}" -u eval_ar.py \
    --data-root "${DATA_ROOT}" \
    --n-inits 20 --n-samples 10 --n-steps 120 \
    --noise-scale "${NOISE_SCALE}" \
    --tag "${TAG}" \
    --chunk-size 50
