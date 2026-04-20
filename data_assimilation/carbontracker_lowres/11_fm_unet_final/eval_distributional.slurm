#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=fm11_eval
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/11_fm_unet_final/slurm_eval_%j.out

source activate neuraltransport
export PYTHONFAULTHANDLER=1

OUT_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/11_fm_unet_final

srun python3 -u "${OUT_DIR}/train_best.py" \
    --study-db "${OUT_DIR}/optuna_study.db" \
    --only-pred --ckpt best
