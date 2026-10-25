#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Submitting CARP-S BBSubset Held-Out Test Sweep Part 2 (3,000 tasks)"
echo "Optimizers: Proximity AC CV, Proximity B, Proximity B CV, Proximity BC, Proximity BC CV"
echo "=================================================="

RESULTS_DIR="results/sweep_bbsubset_test"
TASK_FILE="${RESULTS_DIR}/tasks_part2.txt"

DRY_RUN=false
for arg in "$@"; do
    if [ "$arg" == "--dry-run" ]; then
        DRY_RUN=true
    fi
done

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "Task file ${TASK_FILE} missing or empty. Generating tasks..."
    if [ -f "/bigwork/nhwpseis/.conda/envs/dyrf/bin/python" ]; then
        /bigwork/nhwpseis/.conda/envs/dyrf/bin/python scripts/generate_bbsubset_test_tasks.py
    elif [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_bbsubset_test_tasks.py
    else
        python3 scripts/generate_bbsubset_test_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
echo "Total tasks in Part 2: ${TOTAL_TASKS}"

mkdir -p "${RESULTS_DIR}/logs"

CHUNK_SIZE=200
CONCURRENCY=25

echo "Submitting ${TOTAL_TASKS} tasks in chunks of ${CHUNK_SIZE} (concurrency %${CONCURRENCY})..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] sbatch --job-name=bb_test_p2 --output=${RESULTS_DIR}/logs/bb_test_p2_%A_%a.log --error=${RESULTS_DIR}/logs/bb_test_p2_%A_%a.err --export=ALL,TASK_FILE=${TASK_FILE} --array=${start}-${end}%${CONCURRENCY} scripts/submit_bbsubset_test_array.sbatch"
    else
        JOB_ID=$(sbatch --parsable --job-name=bb_test_p2 --output=${RESULTS_DIR}/logs/bb_test_p2_%A_%a.log --error=${RESULTS_DIR}/logs/bb_test_p2_%A_%a.err --export=ALL,TASK_FILE=${TASK_FILE} --array=${start}-${end}%${CONCURRENCY} scripts/submit_bbsubset_test_array.sbatch)
        echo "Submitted Part 2 Chunk (${start}-${end} / ${TOTAL_TASKS}) -> Job ID: ${JOB_ID}"
    fi
done

echo "=================================================="
if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed for Part 2. All commands verified."
else
    echo "All ${TOTAL_TASKS} tasks of Part 2 successfully scheduled on LUIS cluster!"
fi
echo "=================================================="
