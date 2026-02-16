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
#SBATCH -o /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data_assimilation/carbontracker_midres/01_fm_unet_training_baseline/slurm_%A.out

# --- Environment setup ---
source activate neuraltransport
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting run at: $(date)"
printenv
echo "TMDIR: $TEMPDIR"

# --- Scratch setup ---
# Create a unique local scratch directory per job
LOCAL_DIR="/scratch/jgross/${SLURM_JOB_ID:?}/Carbontracker"
REMOTE_DIR="/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?}"

mkdir -p "${LOCAL_DIR}"

# Cleanup function to remove scratch even on failure/cancel
function clean_up {
    echo "Cleaning up local scratch at $(date)"
    cd "${SUBMIT_DIR}" || exit
    rm -rf "${LOCAL_DIR:?}"
    echo "Cleanup done."
    exit
}

# Call clean_up on any exit (success, failure, cancel)
trap 'clean_up' EXIT

# --- Copy data to scratch ---
for split in train val
do
    mkdir -p "${LOCAL_DIR}/${split}"
    for file in carbontracker_latlon2.8125_l20_6h.zarr carbontracker_latlon2.8125_l20_6h_stats.zarr
    do
        echo "Copying ${split} ${file}"
        cp -r "${REMOTE_DIR}/${split}/${file}" "${LOCAL_DIR}/${split}/${file}"
    done
done

# --- Run flow matching training ---
srun python3 -u /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/data_assimilation/carbontracker_midres/01_fm_unet_training_baseline/train.py --data_path "${LOCAL_DIR}" # --rollout --only_pred --ckpt "best"