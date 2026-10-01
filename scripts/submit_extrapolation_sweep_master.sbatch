#!/bin/bash
# ==============================================================================
# Master Slurm Orchestrator for 9,600-Task Extrapolation UQ Sweep
# ------------------------------------------------------------------------------
# LUIS limits total user jobs to 5,000 at any time.
# To safely execute all 9,600 tasks without exceeding the 5,000 job limit:
#   Stage 1: Submit tasks 1 to 4,800 (24 array jobs of 200 tasks each, %25 concurrency)
#   Watcher: Monitor squeue --me until only this master job remains
#   Stage 2: Submit tasks 4,801 to 9,600 (24 array jobs of 200 tasks each, %25 concurrency)
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

echo "=================================================="
echo "Master Orchestrator Job ID: ${SLURM_JOB_ID}"
echo "Running on Host: $(hostname)"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# 1. Ensure tasks file is generated and contains 9,600 tasks
TASK_FILE="${TASK_FILE:-results/extrapolation_sweep_tasks.txt}"
if [ ! -f "$TASK_FILE" ] || [ "$(wc -l < "$TASK_FILE" | tr -d ' ')" -ne 9600 ]; then
    echo "Generating full 9,600 sweep tasks file: ${TASK_FILE}..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_extrapolation_sweep_tasks.py --output-file "$TASK_FILE"
    else
        python3 scripts/generate_extrapolation_sweep_tasks.py --output-file "$TASK_FILE"
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
echo "Verified task count: ${TOTAL_TASKS} tasks"

if [ "$TOTAL_TASKS" -ne 9600 ]; then
    echo "WARNING: Expected 9,600 tasks, found ${TOTAL_TASKS}." >&2
fi

STAGE1_START=1
STAGE1_END=4800
STAGE2_START=4801
STAGE2_END=$TOTAL_TASKS

# ------------------------------------------------------------------------------
# STAGE 1: Tasks 1 to 4,800
# ------------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 1: Dispatching tasks ${STAGE1_START} to ${STAGE1_END}..."
echo "=================================================="

START_TASK=$STAGE1_START END_TASK=$STAGE1_END bash scripts/submit_extrapolation_sweep_all.sh

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1 dispatched. Monitoring queue..."

while true; do
    if command -v squeue &>/dev/null; then
        # Count user jobs excluding this master job
        ACTIVE_JOBS=$(squeue --me -h 2>/dev/null | awk '{print $1}' | grep -v "^${SLURM_JOB_ID:-NONEXISTENT}" | wc -l | tr -d ' ')
    else
        ACTIVE_JOBS=0
    fi

    if [ "$ACTIVE_JOBS" -le 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1 complete! Only master job remains in queue."
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 1 in progress: ${ACTIVE_JOBS} active/pending jobs remaining. Sleeping 60s..."
    sleep 60
done

# ------------------------------------------------------------------------------
# STAGE 2: Tasks 4,801 to 9,600
# ------------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STAGE 2: Dispatching tasks ${STAGE2_START} to ${STAGE2_END}..."
echo "=================================================="

START_TASK=$STAGE2_START END_TASK=$STAGE2_END bash scripts/submit_extrapolation_sweep_all.sh

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2 dispatched. Monitoring queue..."

while true; do
    if command -v squeue &>/dev/null; then
        ACTIVE_JOBS=$(squeue --me -h 2>/dev/null | awk '{print $1}' | grep -v "^${SLURM_JOB_ID:-NONEXISTENT}" | wc -l | tr -d ' ')
    else
        ACTIVE_JOBS=0
    fi

    if [ "$ACTIVE_JOBS" -le 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2 complete! All sweep array jobs finished."
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stage 2 in progress: ${ACTIVE_JOBS} active/pending jobs remaining. Sleeping 60s..."
    sleep 60
done

# ------------------------------------------------------------------------------
# POST-PROCESSING: Automated Results Aggregation
# ------------------------------------------------------------------------------
echo ""
echo "=================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running automated downstream calibration scorecard aggregation..."
echo "=================================================="

if [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/aggregate_extrapolation_results.py
else
    python3 scripts/aggregate_extrapolation_results.py
fi

echo "=================================================="
echo "All 9,600 tasks completed and aggregated successfully!"
echo "Finish Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
