#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Preparing Unweighted Proximity Sweep"
echo "=================================================="

# 1. Resolve Python interpreter
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

# 2. Generate Task File if missing or requested
TASK_FILE="${TASK_FILE:-results/unweighted_proximity_sweep/tasks.txt}"
SWEEP_MODE="${SWEEP_MODE:-full}"

if [ ! -f "$TASK_FILE" ]; then
    echo "Task file ${TASK_FILE} missing. Generating tasks with mode '${SWEEP_MODE}'..."
    "$PYTHON_BIN" scripts/generate_unweighted_proximity_sweep_tasks.py --mode "$SWEEP_MODE" --output-file "$TASK_FILE"
fi

TOTAL_TASKS=$(grep -c "^[^\#]" "$TASK_FILE" || true)
echo "Total tasks in task file: $TOTAL_TASKS"

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/unweighted_proximity_sweep/logs
mkdir -p results/unweighted_proximity_sweep/raw
mkdir -p results/unweighted_proximity_sweep/summaries

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
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%${CONCURRENCY} scripts/submit_unweighted_proximity_sweep_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "Tasks ${START_TASK} to ${END_TASK} successfully scheduled on cluster!"
echo "=================================================="
