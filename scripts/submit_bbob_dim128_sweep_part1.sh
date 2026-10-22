#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Submitting BBOB Dim 128 Sweep - Part 1 (Tasks 1 to 3,600)"
echo "=================================================="

RESULTS_DIR="results/sweep_bbob_dim128"
TASK_FILE="${RESULTS_DIR}/tasks_part1.txt"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "Task file $TASK_FILE missing or empty. Generating tasks..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_bbob_dim128_sweep_tasks.py
    else
        python3 scripts/generate_bbob_dim128_sweep_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
echo "Submitting Part 1 array: 1-${TOTAL_TASKS}%50 using task file: $TASK_FILE"

TASK_FILE="$TASK_FILE" sbatch \
    --array=1-3600%50 \
    --job-name=b128_p1 \
    --export=ALL,TASK_FILE="$TASK_FILE" \
    scripts/submit_bbob_dim128_sweep_array.sbatch

echo "Part 1 submitted successfully."
