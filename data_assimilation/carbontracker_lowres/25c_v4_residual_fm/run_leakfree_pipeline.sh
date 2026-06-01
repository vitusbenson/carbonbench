#!/bin/bash
# P2: submit the leak-free training pipeline as a SLURM dependency chain.
#   stage 1: single-step backbone  (phase1_det_backbone_leakfree)
#   stage 2: rollout-FT + EMA freeze (phase1p_det_stable_rollout_ft_leakfree)
#   stage 3: sigma_res + residual-FM head (phase2p_residual_fm_stable_leakfree)
# Each stage warm-starts from the previous; stage N+1 runs afterok:N.
set -euo pipefail
ROOT=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm

j1=$(sbatch --parsable "${ROOT}/phase1_det_backbone_leakfree/train.slurm")
echo "stage1 (backbone)      = ${j1}"
j2=$(sbatch --parsable --dependency=afterok:${j1} "${ROOT}/phase1p_det_stable_rollout_ft_leakfree/train_uniform.slurm")
echo "stage2 (rollout-FT)    = ${j2}  (afterok:${j1})"
j3=$(sbatch --parsable --dependency=afterok:${j2} "${ROOT}/phase2p_residual_fm_stable_leakfree/train.slurm")
echo "stage3 (residual-FM)   = ${j3}  (afterok:${j2})"
echo "Submitted chain: ${j1} -> ${j2} -> ${j3}"
