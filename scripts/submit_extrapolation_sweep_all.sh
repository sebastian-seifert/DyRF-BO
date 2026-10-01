#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Preparing Extrapolation Uncertainty Quantification Sweep"
echo "=================================================="

# 1. Generate Task File if missing
TASK_FILE="${TASK_FILE:-results/extrapolation_sweep_tasks.txt}"
if [ ! -f "$TASK_FILE" ]; then
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_extrapolation_sweep_tasks.py
    else
        python3 scripts/generate_extrapolation_sweep_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
echo "Total tasks in task file: $TOTAL_TASKS"

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/extrapolation_uq/logs

# LUIS MaxArraySize limit is typically 200 or 1000. Chunking into 200 tasks per sub-array (<= 300 limit):
CHUNK_SIZE=200
CONCURRENCY=25

START_TASK="${START_TASK:-1}"
END_TASK="${END_TASK:-$TOTAL_TASKS}"

if [ "$END_TASK" -gt "$TOTAL_TASKS" ]; then
    END_TASK=$TOTAL_TASKS
fi

echo "Submitting tasks ${START_TASK} to ${END_TASK} (out of ${TOTAL_TASKS}) in chunks of ${CHUNK_SIZE} (concurrency %${CONCURRENCY})..."

for (( start=START_TASK; start<=END_TASK; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $END_TASK ]; then
        end=$END_TASK
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%${CONCURRENCY} scripts/submit_extrapolation_sweep_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "Tasks ${START_TASK} to ${END_TASK} successfully scheduled on cluster!"
echo "=================================================="
