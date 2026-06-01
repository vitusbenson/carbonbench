#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p2_lf_residualfm
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2p_residual_fm_stable_leakfree/slurm_%x_%j.out

# P2 leak-free residual-FM head: recompute sigma_res on the leak-free EMA backbone,
# then train the FM residual head (20k steps, bs=128, no EMA).

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PATH=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin:$PATH
ROOT=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm
EXP_DIR=${ROOT}/phase2p_residual_fm_stable_leakfree
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker_leakfree
DET_CKPT=${ROOT}/phase1p_det_stable_rollout_ft_leakfree/singlestep/checkpoints/ema_frozen.ckpt
SIGMA_RES=${EXP_DIR}/sigma_res_leakfree.npy

MAX_STEPS=${MAX_STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-128}

cd "${EXP_DIR}"
# 1) Recompute sigma_res from the leak-free EMA backbone residuals.
srun "${PYBIN}" -u compute_sigma_res.py \
    --det-ckpt "${DET_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --output "${SIGMA_RES}"

# 2) Train the residual FM head.
srun "${PYBIN}" -u train.py \
    --training-data-root "${DATA_ROOT}" \
    --det-ckpt "${DET_CKPT}" \
    --sigma-res "${SIGMA_RES}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}"
