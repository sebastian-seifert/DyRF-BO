#!/bin/bash
set -e

echo "=================================================="
echo "Preparing Standard 1D-15D Synthetic Benchmark CPU Array Job"
echo "=================================================="

TASK_FILE="results/standard_sweep_tasks.txt"

if [ ! -f "$TASK_FILE" ]; then
    echo "Generating task file..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_standard_sweep_tasks.py
    else
        python3 scripts/generate_standard_sweep_tasks.py
    fi
else
    echo "Using existing task file: ${TASK_FILE}"
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/standard_sweep/logs

CHUNK_SIZE=200
echo "Submitting ${TOTAL_TASKS} tasks for Standard 1D-15D Sweep in chunks of ${CHUNK_SIZE} (LUIS MaxArraySize compliant)..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_standard_sweep_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "Standard 1D-15D Sweep successfully scheduled on LUIS cluster!"
echo "Parse results when finished: python scripts/parse_standard_sweep_results.py"
echo "=================================================="
