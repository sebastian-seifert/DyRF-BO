#!/bin/bash
# ==============================================================================
# Master Slurm Orchestrator for 23,040-Task Extrapolation UQ Sweep (23040 tasks)
# ------------------------------------------------------------------------------
# LUIS limits total user jobs to 5,000 at any time.
# To safely execute all 23,040 tasks without exceeding the 5,000 job limit,
# tasks are partitioned into 6 stages of 3,840 tasks each:
#   Stage 1: Submit tasks 1 to 3840 (20 array jobs of <=200 tasks each, %25 concurrency)
#   Stage 2: Submit tasks 3841 to 7680
#   Stage 3: Submit tasks 7681 to 11520
#   Stage 4: Submit tasks 11521 to 15360
#   Stage 5: Submit tasks 15361 to 19200
#   Stage 6: Submit tasks 19201 to 23040
#   Watcher: Monitor squeue --me between stages until only this master job ($SLURM_JOB_ID) remains
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
