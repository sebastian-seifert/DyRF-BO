#!/bin/bash
set -e

echo "=================================================="
echo "Preparing CARP-S BBOB Dimension 2 Benchmark Sweep"
echo "=================================================="

RESULTS_DIR="results/sweep_bbob_dim2"
TASK_FILE="${RESULTS_DIR}/tasks.txt"

DRY_RUN=false
for arg in "$@"; do
    if [ "$arg" == "--dry-run" ]; then
        DRY_RUN=true
    fi
done

if [ ! -f "$TASK_FILE" ]; then
    echo "Generating task manifest for Dimension 2..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_bbob_dim2_sweep_tasks.py
    else
        python3 scripts/generate_bbob_dim2_sweep_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
echo "Total Dimension 2 Tasks: ${TOTAL_TASKS}"
mkdir -p "${RESULTS_DIR}/logs"

CHUNK_SIZE=200
ARRAY_CONCURRENCY=25

echo ""
echo "--- Submitting BBOB Dim 2 Tasks in chunks of ${CHUNK_SIZE} (max concurrency ${ARRAY_CONCURRENCY}) ---"

CHUNK_INDEX=1
for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] sbatch --job-name=bbob_d2_c${CHUNK_INDEX} --output=${RESULTS_DIR}/logs/bbob_d2_${CHUNK_INDEX}_%A_%a.log --error=${RESULTS_DIR}/logs/bbob_d2_${CHUNK_INDEX}_%A_%a.err --export=ALL,TASK_FILE=${TASK_FILE} --array=${start}-${end}%${ARRAY_CONCURRENCY} scripts/submit_bbob_dim2_sweep_array.sbatch"
    else
        JOB_ID=$(sbatch --parsable --job-name="bbob_d2_c${CHUNK_INDEX}" --output="${RESULTS_DIR}/logs/bbob_d2_${CHUNK_INDEX}_%A_%a.log" --error="${RESULTS_DIR}/logs/bbob_d2_${CHUNK_INDEX}_%A_%a.err" --export="ALL,TASK_FILE=${TASK_FILE}" --array="${start}-${end}%${ARRAY_CONCURRENCY}" scripts/submit_bbob_dim2_sweep_array.sbatch)
        echo "Submitted Chunk ${CHUNK_INDEX} (${start}-${end}) -> Job ID: ${JOB_ID}"
    fi
    CHUNK_INDEX=$(( CHUNK_INDEX + 1 ))
done

echo ""
if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Verified ${TOTAL_TASKS} tasks ready for scheduling."
else
    echo "All ${TOTAL_TASKS} Dimension 2 tasks submitted successfully."
fi
