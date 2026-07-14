#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=carps_dyrf_sweep
#SBATCH --output=results/carps_sweep_%j.log
#SBATCH --error=results/carps_sweep_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Dedicated cluster job submitter for CARP-S HPOBench Dynamic BO Sweeps

# Initialize Conda and activate environment if using conda
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

# Limit thread counts to prevent thread/memory exhaustion from parallel dispatches
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ensure results directory exists
mkdir -p results

echo "=================================================="
echo "SUBMITTING CARP-S DYNAMIC BO RUN TO CLUSTER"
echo "Job ID: $SLURM_JOB_ID"
echo "Arguments: $@"
echo "=================================================="

# Execute the run script
./scripts/run_hpobench_carps_sweep.sh "$@"

echo "=================================================="
echo "CLUSTER JOB COMPLETED"
echo "=================================================="
