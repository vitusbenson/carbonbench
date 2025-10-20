#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240000
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=flowmatching_pred
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_20251016_4_sfno_dev/slurm_%A.out


source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting prediction run at: $(date)"
printenv
echo "TMDIR: $TEMPDIR"

# Paths
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

# Define output directory (current dev version)
OUT_DIR=/Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_20251016_4_sfno_dev

# Run model in prediction-only mode
srun python3 -u $OUT_DIR/train.py --data_path $LOCAL_DIR # --rollout, --ckpt "best", --only_pred

# Clean up scratch
rm -rf $LOCAL_DIR

echo "Finished run at: $(date)"
