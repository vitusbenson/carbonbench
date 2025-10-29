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
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_20251029_4_unet_evaluation_dev/slurm_%A.out


source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting prediction run at: $(date)"
printenv
echo "TMDIR: $TEMPDIR"

# Paths
LOCAL_DIR=/scratch/jgross/$SLURM_JOB_ID/Carbontracker/
REMOTE_DIR=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/

# Copy validation data only (no need for train data if only predicting)
mkdir -p $LOCAL_DIR/val
for file in carbontracker_latlon5.625_l10_6h.zarr carbontracker_latlon5.625_l10_6h_stats.zarr
do
    echo "Copying val $file"
    cp -r $REMOTE_DIR/val/$file $LOCAL_DIR/val/$file
done

# Define output directory (current dev version)
OUT_DIR=/Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_20251029_4_unet_evaluation_dev

# Run model in prediction-only mode
srun python3 -u $OUT_DIR/train.py --data_path $LOCAL_DIR --only_pred # --rollout, --ckpt "best"

# Clean up scratch
rm -rf $LOCAL_DIR

echo "Finished prediction run at: $(date)"
