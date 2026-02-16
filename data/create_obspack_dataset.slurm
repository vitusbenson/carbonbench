#!/bin/bash
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240000
#SBATCH --job-name=obspack_create_dataset
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data/Obspack_create_dataset_%A.out

source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting Obspack dataset creation job at $(date)"
printenv

# Set variables
SAVE_DIR="/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/"
FREQ="3h"
echo "Frequency: $FREQ"

SCRIPT_DIR=/Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data

# Run the pipeline
srun python3 -u  $SCRIPT_DIR/create_obspack_dataset.py \
    --save_dir "$SAVE_DIR" \
    --freq "$FREQ"

echo "Finished Obspack dataset creation at: $(date)"