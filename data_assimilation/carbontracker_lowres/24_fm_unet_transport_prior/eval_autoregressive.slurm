#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=fm24_eval_ar
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/24_fm_unet_transport_prior/slurm_eval_%j.out

source activate neuraltransport

export PYTHONFAULTHANDLER=1

OUT_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/24_fm_unet_transport_prior

# Ensemble auto-regressive: 10 inits × 10 samples × up-to-1yr trajectories
srun python3 -u "${OUT_DIR}/eval_autoregressive.py" \
    --n-init-points 10 --n-samples 10 --n-steps full

# Sliding-window (re-init every 120 steps ≈ 30 days at 6h freq)
srun python3 -u "${OUT_DIR}/eval_autoregressive.py" \
    --n-init-points 10 --n-samples 10 --n-steps full --reinit-every 120
