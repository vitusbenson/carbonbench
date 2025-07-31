#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240000
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=flowmatching
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_firstrun_20250730_dev/slurm_%A.out

source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting run at: `date`"
printenv
echo "TMDIR: $TEMPDIR"

# Local scratch paths
LOCAL_DIR=/scratch/jgross/$SLURM_JOB_ID/Carbontracker/
REMOTE_DIR=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/

# Copy data (training and validation)
for split in train val
do
    mkdir -p $LOCAL_DIR/$split
    for file in carbontracker_latlon5.625_l10_6h.zarr carbontracker_latlon5.625_l10_6h_stats.zarr 
    do
        echo "Copying $split $file"
        cp -r $REMOTE_DIR/$split/$file $LOCAL_DIR/$split/$file
    done
done

# Run flow matching training
srun python3 -u /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_firstrun_20250730_dev/train.py --data_path $LOCAL_DIR # --rollout --only_pred --ckpt "best"

# Clean up scratch
rm -rf $LOCAL_DIR
