#!/bin/bash
# Submit Sweep 5 (Manifold OOD) in dependent 300-task chunks
set -e

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

SWEEP_DIR="results/sweep_5_manifold"
mkdir -p "${SWEEP_DIR}/logs"

echo "Generating Sweep 5 array tasks..."
python3 scripts/generate_sweep5_tasks.py

TOTAL_TASKS=$(wc -l < "${SWEEP_DIR}/tasks.txt")
CHUNK_SIZE=300

PARENT_DEP=$1

echo "Submitting ${TOTAL_TASKS} tasks for Sweep 5 in chunks of ${CHUNK_SIZE}..."

PREV_JOB=""
for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi

    DEP_FLAG=""
    if [ -n "$PREV_JOB" ]; then
        DEP_FLAG="--dependency=afterany:${PREV_JOB}"
    elif [ -n "$PARENT_DEP" ]; then
        DEP_FLAG="--dependency=afterany:${PARENT_DEP}"
    fi

    if [ -n "$DEP_FLAG" ]; then
        JOB_ID=$(sbatch --parsable ${DEP_FLAG} --job-name=sweep_5_manifold --output="${SWEEP_DIR}/logs/array_%A_%a.log" --error="${SWEEP_DIR}/logs/array_%A_%a.err" --array=${start}-${end}%15 scripts/submit_synthetic_array.sbatch "$SWEEP_DIR")
        echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID} (${DEP_FLAG})"
    else
        JOB_ID=$(sbatch --parsable --job-name=sweep_5_manifold --output="${SWEEP_DIR}/logs/array_%A_%a.log" --error="${SWEEP_DIR}/logs/array_%A_%a.err" --array=${start}-${end}%15 scripts/submit_synthetic_array.sbatch "$SWEEP_DIR")
        echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
    fi
    PREV_JOB=$JOB_ID
done

echo "--------------------------------------------------------"
echo "Sweep 5 successfully scheduled."
echo "Admin progress logging command:"
echo "python3 scripts/monitor_progress.py ${SWEEP_DIR}"
echo "LAST_JOB_ID:${PREV_JOB}"
echo "--------------------------------------------------------"
