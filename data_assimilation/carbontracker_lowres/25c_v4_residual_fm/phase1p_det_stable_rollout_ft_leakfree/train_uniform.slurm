#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p2_lf_rolloutft
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase1p_det_stable_rollout_ft_leakfree/slurm_%x_%j.out

# P2 leak-free rollout-FT: uniform K=4, lr_mult=0.1, EMA(0.999,start=200), 16k steps.
# Warm-start from the leak-free single-step backbone. Then freeze EMA weights.

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PATH=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin:$PATH
ROOT=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm
EXP_DIR=${ROOT}/phase1p_det_stable_rollout_ft_leakfree
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker_leakfree
P1_CKPT=${ROOT}/phase1_det_backbone_leakfree/singlestep/checkpoints/best.ckpt

cd "${EXP_DIR}"
srun "${PYBIN}" -u train.py \
    --training-data-root "${DATA_ROOT}" \
    --max-steps 16000 \
    --batch-size 32 \
    --rollout-K 4 \
    --ckpt-from "${P1_CKPT}" \
    --lr-mult 0.1 \
    --no-step-discount \
    --ema-decay 0.999 \
    --ema-start 200

# Freeze EMA shadow weights -> ema_frozen.ckpt for the residual-FM stage.
"${PYBIN}" -u "${ROOT}/freeze_ema_ckpt.py" \
    --in  "${EXP_DIR}/singlestep/checkpoints/last.ckpt" \
    --out "${EXP_DIR}/singlestep/checkpoints/ema_frozen.ckpt"
