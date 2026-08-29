#!/bin/bash
set -e

CHUNK_SIZE=200
TASK_FILE="results/sweep_noisy_ei_head_to_head/tasks.txt"
SBATCH_FILE="scripts/submit_noisy_ei_sweep_array.sbatch"

# 1. Ensure tasks are generated
if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "[!] Task file missing or empty. Auto-generating tasks..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_noisy_sweep_tasks.py
    else
        python3 scripts/generate_noisy_sweep_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
echo "=================================================="
echo "Submitting Noisy EI Head-to-Head Sweep"
echo "Total Tasks: ${TOTAL_TASKS} | Chunk Size: ${CHUNK_SIZE}"
echo "Task File:   ${TASK_FILE}"
echo "SBATCH File: ${SBATCH_FILE}"
echo "=================================================="

mkdir -p results/sweep_noisy_ei_head_to_head/logs
mkdir -p results/sweep_noisy_ei_head_to_head/runs

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 "$SBATCH_FILE")
    echo "  -> Submitted Chunk [${start} - ${end}] | Array Job ID: ${JOB_ID}"
done

echo ""
echo "=================================================="
echo "[✓] All 3,840 tasks scheduled across $(( (TOTAL_TASKS + CHUNK_SIZE - 1) / CHUNK_SIZE )) chunk arrays!"
echo "=================================================="
