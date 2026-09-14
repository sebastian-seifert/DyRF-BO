#!/bin/bash
set -e

echo "=================================================="
echo "Preparing CARP-S BBsubset Proximity Meta-HPO Sweep"
echo "=================================================="

TASK_FILE="results/sweep_proximity_meta_hpo/tasks.txt"
CONFIG_FILE="results/sweep_proximity_meta_hpo/sobol_configs.json"

if [ -f "$TASK_FILE" ] && [ -s "$TASK_FILE" ]; then
    echo "Using existing task file: ${TASK_FILE}"
else
    echo "Task file missing. Generating from Sobol configurations..."
    if [ ! -f "$CONFIG_FILE" ] || [ ! -s "$CONFIG_FILE" ]; then
        echo "Sampling Sobol configurations..."
        if [ -f ".venv/bin/python" ]; then
            .venv/bin/python scripts/sample_proximity_meta_configs.py -o "$CONFIG_FILE"
        else
            python3 scripts/sample_proximity_meta_configs.py -o "$CONFIG_FILE"
        fi
    fi

    echo "Generating task file..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_proximity_meta_sweep_tasks.py -o "$TASK_FILE" -c "$CONFIG_FILE"
    else
        python3 scripts/generate_proximity_meta_sweep_tasks.py -o "$TASK_FILE" -c "$CONFIG_FILE"
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/sweep_proximity_meta_hpo/logs

CHUNK_SIZE=200
echo "Submitting ${TOTAL_TASKS} tasks for Proximity Meta-HPO Sweep in chunks of ${CHUNK_SIZE} (LUIS MaxArraySize <= 200, %25 concurrency)..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_proximity_meta_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "Proximity Meta-HPO Sweep successfully scheduled on LUIS cluster!"
echo "=================================================="
