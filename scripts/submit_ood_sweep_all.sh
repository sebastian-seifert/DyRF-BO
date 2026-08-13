#!/bin/bash
set -e

echo "=================================================="
echo "Preparing Synthetic OOD Benchmark CPU Array Job"
echo "=================================================="

# 1. Generate Task File (630 tasks)
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/generate_ood_sweep_tasks.py
else
    python3 scripts/generate_ood_sweep_tasks.py
fi

TASK_FILE="results/ood_sweep_tasks.txt"
TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/ood_sweep/logs

CHUNK_SIZE=200
echo "Submitting ${TOTAL_TASKS} tasks for Synthetic OOD Sweep in chunks of ${CHUNK_SIZE} (LUIS MaxArraySize compliant)..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_ood_sweep_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "Synthetic OOD Sweep successfully scheduled on LUIS cluster!"
echo "Parse results when finished: python scripts/parse_ood_sweep_results.py"
echo "=================================================="
