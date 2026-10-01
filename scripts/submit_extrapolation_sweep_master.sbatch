#!/bin/bash
# ==============================================================================
# Master Slurm Orchestrator for 9,600-Task Extrapolation UQ Sweep
# ------------------------------------------------------------------------------
# LUIS limits total user jobs to 5,000 at any time.
# To safely execute all 9,600 tasks without exceeding the 5,000 job limit:
#   Stage 1: Submit tasks 1 to 4800 (24 array jobs of 200 tasks each, %25 concurrency)
#   Watcher: Monitor squeue --me until only this master job ($SLURM_JOB_ID) remains
#   Stage 2: Submit tasks 4801 to 9600 (24 array jobs of 200 tasks each, %25 concurrency)
#   Watcher: Monitor squeue --me until all tasks finish
#   Post-processing: Automatically execute aggregate_extrapolation_results.py
# ==============================================================================
#SBATCH -p ai
#SBATCH --job-name=uq_sweep_master
#SBATCH --output=results/extrapolation_uq/logs/master_orchestrator_%j.out
#SBATCH --error=results/extrapolation_uq/logs/master_orchestrator_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results/extrapolation_uq/logs

if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" scripts/run_extrapolation_sweep_orchestrator.py "$@"
