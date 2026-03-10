#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240000
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=fm_ot_training
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data_assimilation/carbontracker_lowres/09_fm_unet_ot_training/slurm_%A.out


source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting run at: $(date)"
printenv
echo "TMDIR: $TEMPDIR"

# --- Scratch setup ---
LOCAL_DIR="/scratch/jgross/${SLURM_JOB_ID:?}/Carbontracker"
REMOTE_DIR="/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?}"

mkdir -p "${LOCAL_DIR}"

function clean_up {
    echo "Cleaning up local scratch at $(date)"
    cd "${SUBMIT_DIR}" || exit
    rm -rf "${LOCAL_DIR:?}"
    echo "Cleanup done."
    exit
}

trap 'clean_up' EXIT

# --- Copy data (training and validation) ---
for split in train val
do
    mkdir -p "${LOCAL_DIR}/${split}"
    for file in carbontracker_latlon5.625_l10_6h.zarr carbontracker_latlon5.625_l10_6h_stats.zarr
    do
        echo "Copying ${split} ${file}"
        cp -r "${REMOTE_DIR}/${split}/${file}" "${LOCAL_DIR}/${split}/${file}"
    done
done

# Define output directory
OUT_DIR=/Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data_assimilation/carbontracker_lowres/09_fm_unet_ot_training

srun python3 -u $OUT_DIR/train.py --training_data_root "${LOCAL_DIR}"

echo "Finished run at: $(date)"
