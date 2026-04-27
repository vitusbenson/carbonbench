#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p25g_p2b_residual_rollout
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2b_residual_fm_rollout_ft/slurm_%x_%j.out

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PATH=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin:$PATH
EXP_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2b_residual_fm_rollout_ft
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker

MAX_STEPS=${MAX_STEPS:-8000}
BATCH_SIZE=${BATCH_SIZE:-32}
ROLLOUT_K=${ROLLOUT_K:-4}
LR_MULT=${LR_MULT:-0.1}

cd "${EXP_DIR}"
srun "${PYBIN}" -u train.py \
    --training-data-root "${DATA_ROOT}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --rollout-K "${ROLLOUT_K}" \
    --lr-mult "${LR_MULT}"
