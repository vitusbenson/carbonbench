#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=fm_optuna
#SBATCH --mail-type=END
#SBATCH --array=0-31%16
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/18_fm_tuning/slurm_%A_%a.out

source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting worker ${SLURM_ARRAY_TASK_ID} at: $(date)"

# --- Scratch setup ---
LOCAL_DIR="/scratch/vbenson/${SLURM_JOB_ID:?}/Carbontracker"
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

# --- Copy data to local scratch ---
for split in train val test
do
    mkdir -p "${LOCAL_DIR}/${split}"
    for file in carbontracker_latlon5.625_l10_6h.zarr carbontracker_latlon5.625_l10_6h_stats.zarr obs_carbontracker_latlon5.625_l10_6h.zarr
    do
        if [ -d "${REMOTE_DIR}/${split}/${file}" ]; then
            cp -r "${REMOTE_DIR}/${split}/${file}" "${LOCAL_DIR}/${split}/${file}"
        fi
    done
done

OUT_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/18_fm_tuning

# Each worker runs 7 trials (~30min each + 30min data load = ~4h, within 6h limit)
# 32 workers × 7 = 224 total trials; Optuna coordinates via shared SQLite DB
srun python3 -u "${OUT_DIR}/run_optuna.py" \
    --training-data-root "${LOCAL_DIR}" \
    --n-trials 7 \
    --max-steps 3000

echo "Worker ${SLURM_ARRAY_TASK_ID} finished at: $(date)"
