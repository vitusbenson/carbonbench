#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240000
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=posterior_tuning
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/13_posterior_conditioning_ablation/slurm_optuna_%A.out


source activate neuraltransport

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting posterior sampler Optuna tuning at: $(date)"
printenv
echo "TMPDIR: $TEMPDIR"

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

# --- Copy test data ---
for split in test
do
    mkdir -p "${LOCAL_DIR}/${split}"
    for file in carbontracker_latlon5.625_l10_6h.zarr carbontracker_latlon5.625_l10_6h_stats.zarr obs_carbontracker_latlon5.625_l10_6h.zarr
    do
        if [ -d "${REMOTE_DIR}/${split}/${file}" ]; then
            echo "Copying ${split} ${file}"
            cp -r "${REMOTE_DIR}/${split}/${file}" "${LOCAL_DIR}/${split}/${file}"
        fi
    done
done

OUT_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/13_posterior_conditioning_ablation

# Run Optuna tuning for all methods
srun python3 -u $OUT_DIR/run_optuna.py --device cuda --n-trials 50 --n-targets 10 --n-samples 20

echo "Finished posterior sampler Optuna tuning at: $(date)"
