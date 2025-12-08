#!/bin/bash
#SBATCH --job-name=batch_opt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=08:00:00
#SBATCH --output=../output_data/logs/%x_%A.out

# ==========================================================
# Usage:
#   sbatch run_optimisation.slurm SCRIPT_NAME CASE_NAME PARAM_INDICES LOWER_BOUNDS UPPER_BOUNDS TOPOLOGY
#
# Example:
#   sbatch run_optimisation.slurm RUN_batch_culture_n_dim_optimisation.py \
#       test_run \
#       2,3,14 \
#       0,0,0.1 \
#       100,100,2.0 \
#       1,0,0,0,1,0
# ==========================================================

mkdir -p ../output_data/logs

SCRIPT_NAME=$1
CASE_NAME=$2
PARAM_INDICES=$3
LOWER_BOUNDS=$4
UPPER_BOUNDS=$5
TOPOLOGY=$6

OUTDIR="../output_data"
FINAL_FILE="${OUTDIR}/${CASE_NAME}_pareto_results.npz"
PARTIAL_FILE="${OUTDIR}/${CASE_NAME}_pareto_partial.npz"

if [ -f "$FINAL_FILE" ]; then
  echo "Final output $FINAL_FILE already exists, skipping."
  exit 0
fi

if [ -f "$PARTIAL_FILE" ]; then
  echo "Found partial file: $PARTIAL_FILE — optimisation will resume."
else
  echo "No partial checkpoint found — starting fresh."
fi

module purge
source /home/rawsys/matmfc/essentiality_toxicity_circuit/essentiality_toxicity_circuit_venv/bin/activate

export OMP_NUM_THREADS=1           
export MKL_NUM_THREADS=1        
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export p=$SLURM_CPUS_PER_TASK     

CMD="python $SCRIPT_NAME \
        --case $CASE_NAME \
        --param-indices $PARAM_INDICES \
        --lower-bounds $LOWER_BOUNDS \
        --upper-bounds $UPPER_BOUNDS \
        --topology $TOPOLOGY \
        --n-proc $SLURM_CPUS_PER_TASK \
        --pop-size-ga ${POP_GA:-30} \
        --n-gen-ga   ${NGEN_GA:-60}"

echo "Running command:"
echo "  $CMD"
echo ""

srun $CMD
