#!/bin/bash
#SBATCH --job-name=compare_all
#SBATCH --output=slurm_compare_%A_%a.out
#SBATCH --error=slurm_compare_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-9

# 10 methods: unconditional + 9 posterior methods
METHODS=(unconditional dps flowdps sde fig ictm mcg pcfm fmps dflow)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "Comparison for ${METHOD}: $(date)"
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"

cd /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/13_posterior_conditioning_ablation

python compare_methods.py --method $METHOD --n-targets 20 --n-samples 20 --batch-size 20

echo "Done ${METHOD}: $(date)"
