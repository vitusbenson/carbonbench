#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=fm_compare_11_24
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/compare_11_24/slurm_compare_%j.out

source activate neuraltransport
export PYTHONFAULTHANDLER=1

OUT_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/compare_11_24

# One-step ensemble comparison (20 inits × 10 samples, both models).
srun python3 -u "${OUT_DIR}/eval_compare.py" --mode onestep \
    --n-init-points 20 --n-samples 10

# Trajectory comparison: 60 days (240 steps @ 6h) × 10 inits × 10 samples.
srun python3 -u "${OUT_DIR}/eval_compare.py" --mode trajectory \
    --n-init-points 10 --n-samples 10 --n-steps 240

# Generate comparison plots.
srun python3 -u "${OUT_DIR}/plot_compare.py"
