#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p25g_sigma_res
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2_residual_fm/slurm_%x_%j.out

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PATH=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin:$PATH
EXP_DIR=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase2_residual_fm
DATA_ROOT=/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker
DET_CKPT=/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/25c_v4_residual_fm/phase1b_det_rollout_ft/singlestep/checkpoints/Epoch=13-Step=2716-LossVal=0.920315.ckpt

cd "${EXP_DIR}"
srun "${PYBIN}" -u compute_sigma_res.py \
    --det-ckpt "${DET_CKPT}" \
    --output "${EXP_DIR}/sigma_res.npy" \
    --data-root "${DATA_ROOT}" \
    --batch-size 256 \
    --max-batches 200
