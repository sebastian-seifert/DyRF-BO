#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=ood_sweep_single
#SBATCH --output=results/ood_sweep/logs/ood_sweep_%j.log
#SBATCH --error=results/ood_sweep/logs/ood_sweep_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


mkdir -p results/ood_sweep/logs

echo "=================================================="
echo "Starting Synthetic OOD Benchmark Single SLURM Job"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

python3 scripts/run_ood_sweep_harness.py

echo "=================================================="
echo "OOD Sweep Completed. Parsing Results..."
echo "=================================================="

python3 scripts/parse_ood_sweep_results.py

echo "=================================================="
echo "Finished!"
echo "=================================================="
