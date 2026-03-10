#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=fm_ot_eval
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data_assimilation/carbontracker_lowres/09_fm_unet_ot_training/eval_slurm_%A.out


source activate neuraltransport

export PYTHONFAULTHANDLER=1

echo "Starting evaluation at: $(date)"

OUT_DIR=/Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data_assimilation/carbontracker_lowres/09_fm_unet_ot_training

srun python3 -u $OUT_DIR/run_eval.py --ablation all

echo "Finished evaluation at: $(date)"
