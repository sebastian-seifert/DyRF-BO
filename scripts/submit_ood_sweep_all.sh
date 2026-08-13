#!/bin/bash
set -e

echo "=================================================="
echo "Preparing Synthetic OOD Benchmark CPU Array Job"
echo "=================================================="

# 1. Generate Task File (630 tasks)
python3 scripts/generate_ood_sweep_tasks.py

TASK_FILE="results/ood_sweep_tasks.txt"
TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/ood_sweep/logs

echo "Generated $TOTAL_TASKS tasks in $TASK_FILE."
echo "Submitting SLURM array job with max 50 concurrent tasks..."

JOB_ID=$(sbatch --parsable --array=1-${TOTAL_TASKS}%50 scripts/submit_ood_sweep_array.sbatch)

echo "=================================================="
echo "Submitted SLURM Array Job ID: $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"
echo "Parse results when finished: python scripts/parse_ood_sweep_results.py"
echo "=================================================="
