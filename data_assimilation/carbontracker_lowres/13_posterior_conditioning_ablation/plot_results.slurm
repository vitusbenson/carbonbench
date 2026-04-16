#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --output=slurm_plot_%j.out
#SBATCH --error=slurm_plot_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=work
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "Plotting results: $(date)"

cd /Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/carbonbench/data_assimilation/carbontracker_lowres/13_posterior_conditioning_ablation

python plot_results.py

echo "Done: $(date)"
