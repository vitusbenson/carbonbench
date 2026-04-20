#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=fm24_eval_smoke
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/24_fm_unet_transport_prior/slurm_smoke_%j.out

source activate neuraltransport
export PYTHONFAULTHANDLER=1

OUT_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/24_fm_unet_transport_prior

# Smoke: 2 inits × 2 samples × 20 steps (5 days) trajectory
srun python3 -u "${OUT_DIR}/eval_autoregressive.py" \
    --n-init-points 2 --n-samples 2 --n-steps 20

# Smoke: instantaneous eval, 3 refs × 2 samples
srun python3 -u "${OUT_DIR}/eval_instantaneous.py" \
    --n-ref-timesteps 3 --n-gen-per-ref 2
