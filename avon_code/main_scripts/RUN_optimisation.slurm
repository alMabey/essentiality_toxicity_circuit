#!/bin/bash
#SBATCH --job-name=multiobj_opt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out

mkdir -p logs

module purge
source /home/rawsys/matmfc/essentiality_toxicity_circuit/essentiality_toxicity_circuit_venv/bin/activate

# Run the optimisation
echo "Running optimisation on $SLURM_CPUS_PER_TASK CPUs"

echo "Start time: $(date)"
python RUN_optimisation.py
echo "Finish time: $(date)"
