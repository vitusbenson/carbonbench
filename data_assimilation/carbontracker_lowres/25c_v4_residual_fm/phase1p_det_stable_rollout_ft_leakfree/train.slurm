#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p25p_p1p_rollout
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase1p_det_stable_rollout_ft/slurm_%x_%j.out

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PATH=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin:$PATH
EXP_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase1p_det_stable_rollout_ft
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker
# Use the single-step pretrained Phase 1 best checkpoint (NOT phase1b's diverged one).
P1_CKPT=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase1_det_backbone/singlestep/checkpoints/best.ckpt

MAX_STEPS=${MAX_STEPS:-8000}
BATCH_SIZE=${BATCH_SIZE:-32}
ROLLOUT_K=${ROLLOUT_K:-4}
LR_MULT=${LR_MULT:-0.2}
EMA_DECAY=${EMA_DECAY:-0.999}
EMA_START=${EMA_START:-200}

cd "${EXP_DIR}"
srun "${PYBIN}" -u train.py \
    --training-data-root "${DATA_ROOT}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --rollout-K "${ROLLOUT_K}" \
    --ckpt-from "${P1_CKPT}" \
    --lr-mult "${LR_MULT}" \
    --ema-decay "${EMA_DECAY}" \
    --ema-start "${EMA_START}"
