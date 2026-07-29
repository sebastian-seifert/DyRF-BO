#!/bin/bash
set -e

# Ensure output directories exist
mkdir -p results/epistemic_acq/ei
mkdir -p results/epistemic_acq/pi
mkdir -p results/epistemic_acq/lcb
mkdir -p results/epistemic_acq/baseline/ei
mkdir -p results/epistemic_acq/baseline/pi
mkdir -p results/epistemic_acq/baseline/lcb
mkdir -p results/epistemic_acq/logs

# Ensure task list is generated
if [ ! -f "results/epistemic_full_acq_array_tasks.txt" ]; then
    echo "Generating full acquisition array tasks list..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_epistemic_full_acq_array_tasks.py
    else
        python3 scripts/generate_epistemic_full_acq_array_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < results/epistemic_full_acq_array_tasks.txt)
echo "Submitting Slurm array job for $TOTAL_TASKS tasks..."

# Submit array job with max 25 concurrent tasks
sbatch --array=1-${TOTAL_TASKS}%25 scripts/submit_epistemic_full_acq_array.sbatch
