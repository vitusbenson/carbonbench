#!/bin/bash
#SBATCH --job-name=optuna_all
#SBATCH --output=slurm_optuna_%A_%a.out
#SBATCH --error=slurm_optuna_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-89

# Unified Optuna sweep: 9 methods x 10 workers = 90 array tasks
# Each worker runs 5 trials → 50 trials per method, 450 total
#
# Task mapping:
#   0-9   = dps     (workers 0-9)
#   10-19 = flowdps (workers 0-9)
#   20-29 = sde     (workers 0-9)
#   30-39 = fig     (workers 0-9)
#   40-49 = ictm    (workers 0-9)
#   50-59 = mcg     (workers 0-9)
#   60-69 = pcfm    (workers 0-9)
#   70-79 = fmps    (workers 0-9)
#   80-89 = dflow   (workers 0-9)

METHODS=(dps flowdps sde fig ictm mcg pcfm fmps dflow)
N_WORKERS=10
TRIALS_PER_WORKER=5

METHOD_IDX=$((SLURM_ARRAY_TASK_ID / N_WORKERS))
WORKER_ID=$((SLURM_ARRAY_TASK_ID % N_WORKERS))
METHOD=${METHODS[$METHOD_IDX]}

echo "Optuna worker ${WORKER_ID}/${N_WORKERS} for ${METHOD}: $(date)"
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES, Array task: $SLURM_ARRAY_TASK_ID"

cd /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/13_posterior_conditioning_ablation

python run_optuna.py \
    --method $METHOD \
    --n-trials $TRIALS_PER_WORKER \
    --n-targets 20 \
    --n-samples 20 \
    --worker-id $WORKER_ID

echo "Done worker ${WORKER_ID} for ${METHOD}: $(date)"
