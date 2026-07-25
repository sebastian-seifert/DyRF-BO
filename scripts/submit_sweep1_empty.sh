#!/bin/bash
# Submit Sweep 1 (Empty Hypercube Gap) in dependent 300-task chunks
set -e

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

SWEEP_DIR="results/sweep_1_empty"
mkdir -p "${SWEEP_DIR}/logs"

echo "Generating Sweep 1 array tasks..."
python3 scripts/generate_sweep1_tasks.py

TOTAL_TASKS=$(wc -l < "${SWEEP_DIR}/tasks.txt")
CHUNK_SIZE=300

echo "Submitting ${TOTAL_TASKS} tasks for Sweep 1 in chunks of ${CHUNK_SIZE}..."

PREV_JOB=""
for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi

    if [ -z "$PREV_JOB" ]; then
        JOB_ID=$(sbatch --parsable --job-name=sweep_1_empty --output="${SWEEP_DIR}/logs/array_%A_%a.log" --error="${SWEEP_DIR}/logs/array_%A_%a.err" --array=${start}-${end}%15 scripts/submit_synthetic_array.sbatch "$SWEEP_DIR")
        echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
    else
        JOB_ID=$(sbatch --parsable --dependency=afterany:${PREV_JOB} --job-name=sweep_1_empty --output="${SWEEP_DIR}/logs/array_%A_%a.log" --error="${SWEEP_DIR}/logs/array_%A_%a.err" --array=${start}-${end}%15 scripts/submit_synthetic_array.sbatch "$SWEEP_DIR")
        echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID} (dependent on ${PREV_JOB})"
    fi
    PREV_JOB=$JOB_ID
done

echo "--------------------------------------------------------"
echo "Sweep 1 successfully scheduled."
echo "Admin progress logging command:"
echo "python3 scripts/monitor_progress.py ${SWEEP_DIR}"
echo "--------------------------------------------------------"
