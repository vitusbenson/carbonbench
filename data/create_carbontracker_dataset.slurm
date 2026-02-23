#!/bin/bash
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240000
#SBATCH --job-name=CarbonTracker_create_dataset
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data/CarbonTracker_create_dataset_%A.out

source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting CarbonTracker dataset creation job at $(date)"
printenv

# Set variables
SAVE_DIR="/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/"
GRIDNAME="latlon5.625" # "latlon5.625" or "latlon2.8125" or "latlon2x3"
VERTICAL_LEVELS="l10"  # "l3" or "l10" or "l20" or "l34"
FREQ="6h"  # "3h" or "6h"

echo "Gridname: $GRIDNAME, Vertical Levels: $VERTICAL_LEVELS, Frequency: $FREQ"

SCRIPT_DIR=/Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data

# Run the pipeline
srun python3 -u  $SCRIPT_DIR/create_carbontracker_dataset.py \
    --save_dir "$SAVE_DIR" \
    --gridname "$GRIDNAME" \
    --vertical_levels "$VERTICAL_LEVELS" \
    --freq "$FREQ"

echo "Finished CarbonTracker dataset creation at: $(date)"
