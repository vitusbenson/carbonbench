#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=fm24_tprior
#SBATCH --mail-type=END
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/24_fm_unet_transport_prior/slurm_%j.out

source activate neuraltransport

export PYTHONFAULTHANDLER=1

LOCAL_DIR="/scratch/vbenson/${SLURM_JOB_ID:?}/Carbontracker"
REMOTE_DIR="/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker/"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?}"

mkdir -p "${LOCAL_DIR}"

function clean_up {
    cd "${SUBMIT_DIR}" || exit
    rm -rf "${LOCAL_DIR:?}"
    exit
}
trap 'clean_up' EXIT

for split in train val test; do
    mkdir -p "${LOCAL_DIR}/${split}"
    for file in carbontracker_latlon5.625_l10_6h.zarr carbontracker_latlon5.625_l10_6h_stats.zarr obs_carbontracker_latlon5.625_l10_6h.zarr; do
        if [ -d "${REMOTE_DIR}/${split}/${file}" ]; then
            cp -r "${REMOTE_DIR}/${split}/${file}" "${LOCAL_DIR}/${split}/${file}"
        fi
    done
done

OUT_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/24_fm_unet_transport_prior
PHASE11_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/11_fm_unet_final

srun python3 -u "${OUT_DIR}/train.py" \
    --study-db "${PHASE11_DIR}/optuna_study.db" \
    --training-data-root "${LOCAL_DIR}" \
    --max-steps 10000
