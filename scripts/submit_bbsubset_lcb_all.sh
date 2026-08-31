#!/bin/bash
set -e

echo "=================================================="
echo "Preparing CARP-S BBsubset LCB Benchmark Sweep"
echo "=================================================="

TASK_FILE="results/sweep_bbsubset_lcb/tasks.txt"

# Ensure all directory hierarchies exist
mkdir -p results/bbsubset_lcb/logs
mkdir -p results/bbsubset_lcb/constant
mkdir -p results/bbsubset_lcb/annealed
mkdir -p runs/bbsubset_runs/constant
mkdir -p runs/bbsubset_runs/annealed

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "Generating task file..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_bbsubset_lcb_sweep_tasks.py
    else
        python3 scripts/generate_bbsubset_lcb_sweep_tasks.py
    fi
else
    echo "Using existing task file: ${TASK_FILE}"
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

CHUNK_SIZE=250
echo "Submitting ${TOTAL_TASKS} tasks for CARP-S BBsubset Dual-Schedule LCB Sweep in chunks of ${CHUNK_SIZE} (LUIS MaxArraySize compliant, %25 concurrency)..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_bbsubset_lcb_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "CARP-S BBsubset LCB Sweep successfully scheduled on LUIS cluster!"
echo "=================================================="
