#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p25cv4_p1b_rollout
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase1b_det_rollout_ft/slurm_%x_%j.out

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PATH=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin:$PATH
EXP_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase1b_det_rollout_ft
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker
P1_CKPT=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase1_det_backbone/singlestep/checkpoints/Epoch=96-Step=18818-LossVal=0.346185.ckpt

MAX_STEPS=${MAX_STEPS:-8000}
BATCH_SIZE=${BATCH_SIZE:-32}
ROLLOUT_K=${ROLLOUT_K:-4}
LR_MULT=${LR_MULT:-0.1}

cd "${EXP_DIR}"
srun "${PYBIN}" -u train.py \
    --training-data-root "${DATA_ROOT}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --rollout-only \
    --rollout-K "${ROLLOUT_K}" \
    --ckpt-from "${P1_CKPT}" \
    --lr-mult "${LR_MULT}"
