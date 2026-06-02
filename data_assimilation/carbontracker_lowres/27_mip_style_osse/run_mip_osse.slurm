#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000
#SBATCH --gres=gpu:A40:1
#SBATCH --job-name=p3_mip_osse
#SBATCH -o /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/27_mip_style_osse/slurm_%x_%j.out

# P3: MIP-style OSSE over the leak-free test window (2015-2020).
# Runs the free (no-DA) baseline + the orbit-EnKF on MIP-realistic synthetic
# obs (real OCO-2 footprints + 20-level AKs, P1 forward operator), then the
# realism sweep (AK on/off, sparsity, retrieval noise).

PYBIN=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin/python
export PATH=/Net/Groups/BGI/people/vbenson/miniconda3/envs/neuraltransport/bin:$PATH
cd /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/27_mip_style_osse

NINIT=${NINIT:-20}
NSAMP=${NSAMP:-10}
NSTEPS=${NSTEPS:-120}

# 1) Free (no-DA) baseline.
srun "${PYBIN}" -u eval_mip_osse.py --method none \
    --n-inits ${NINIT} --n-samples ${NSAMP} --n-steps ${NSTEPS} --tag full

# 2) Orbit-EnKF (real AK, full real coverage, no extra retrieval noise).
srun "${PYBIN}" -u eval_mip_osse.py --method enkf \
    --n-inits ${NINIT} --n-samples ${NSAMP} --n-steps ${NSTEPS} --obs-every 4 \
    --enkf-loc-sigma 4 --sigma-obs 0.1 --tag full

# 3) Realism sweep (P3.4): AK ablation, sparsity, retrieval noise.
srun "${PYBIN}" -u eval_mip_osse.py --method enkf \
    --n-inits ${NINIT} --n-samples ${NSAMP} --n-steps ${NSTEPS} --obs-every 4 \
    --enkf-loc-sigma 4 --sigma-obs 0.1 --ak-mode uniform --tag sweep_akuniform

srun "${PYBIN}" -u eval_mip_osse.py --method enkf \
    --n-inits ${NINIT} --n-samples ${NSAMP} --n-steps ${NSTEPS} --obs-every 4 \
    --enkf-loc-sigma 4 --sigma-obs 0.1 --thin-fraction 0.5 --tag sweep_thin50

srun "${PYBIN}" -u eval_mip_osse.py --method enkf \
    --n-inits ${NINIT} --n-samples ${NSAMP} --n-steps ${NSTEPS} --obs-every 4 \
    --enkf-loc-sigma 4 --sigma-obs 0.3 --obs-noise 0.3 --tag sweep_noise03

echo "P3 MIP-style OSSE suite complete."
