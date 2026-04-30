#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p25p_p2c_ema
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2c_residual_fm_ema/slurm_%x_%j.out

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PATH=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin:$PATH
EXP_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2c_residual_fm_ema
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker

MAX_STEPS=${MAX_STEPS:-40000}
BATCH_SIZE=${BATCH_SIZE:-128}
EMA_DECAY=${EMA_DECAY:-0.999}
EMA_START=${EMA_START:-2000}
WARM_START=${WARM_START:-}

EXTRA=""
if [ -n "${WARM_START}" ]; then EXTRA="--warm-start ${WARM_START}"; fi

cd "${EXP_DIR}"
srun "${PYBIN}" -u train.py \
    --training-data-root "${DATA_ROOT}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --ema-decay "${EMA_DECAY}" \
    --ema-start "${EMA_START}" \
    ${EXTRA}
