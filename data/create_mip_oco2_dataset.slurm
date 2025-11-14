#!/bin/bash
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --job-name=OCO2MIP_create_dataset
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data/OCO2MIP_create_dataset_%A.out

source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting OCO2MIP dataset creation job at $(date)"
printenv

# Set variables
SAVE_DIR="/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/"
GRIDNAME="latlon5.625"
VERTICAL_LEVELS="l10"
FREQ="6h"

SCRIPT_DIR=/Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data

# Run the pipeline
srun python3 -u  $SCRIPT_DIR/create_mip_oco2_dataset.py \
    --save_dir "$SAVE_DIR" \
    --gridname "$GRIDNAME" \
    --vertical_levels "$VERTICAL_LEVELS" \
    --freq "$FREQ"

echo "Finished OCO2MIP dataset creation at: $(date)"
