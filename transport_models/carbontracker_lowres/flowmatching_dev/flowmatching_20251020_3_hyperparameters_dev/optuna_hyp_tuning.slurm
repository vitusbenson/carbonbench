#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240000
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=flowmatching_optuna
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_20251020_3_hyperparameters_dev/slurm_%A.out

source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting run at: $(date)"
printenv

# Local scratch
LOCAL_DIR=/scratch/jgross/$SLURM_JOB_ID/Carbontracker/
REMOTE_DIR=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/

# Copy data
for split in train val; do
    mkdir -p $LOCAL_DIR/$split
    for file in carbontracker_latlon5.625_l10_6h.zarr carbontracker_latlon5.625_l10_6h_stats.zarr; do
        echo "Copying $split $file"
        cp -r $REMOTE_DIR/$split/$file $LOCAL_DIR/$split/$file
    done
done

# Run Optuna hyperparameter tuning
srun python3 -u /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_20251020_3_hyperparameters_dev/optuna_hyp_tuning.py

# Clean up scratch
rm -rf $LOCAL_DIR

echo "Finished hyperparameter tuning run at: $(date)"

