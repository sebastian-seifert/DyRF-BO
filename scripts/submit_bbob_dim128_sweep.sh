#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Preparing CARP-S BBOB Dimension 128 Sweep"
echo "=================================================="

# 1. Resolve Python interpreter
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

# 2. Generate Task File if missing
TASK_FILE="${TASK_FILE:-results/sweep_bbob_dim128/tasks.txt}"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "Task file ${TASK_FILE} missing or empty. Generating tasks..."
    "$PYTHON_BIN" scripts/generate_bbob_dim128_sweep_tasks.py
fi

TOTAL_TASKS=$(grep -c "^[^\#]" "$TASK_FILE" || true)
echo "Total tasks in task file: $TOTAL_TASKS"

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p results/sweep_bbob_dim128/logs

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
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%${CONCURRENCY} scripts/submit_bbob_dim128_sweep_array.sbatch)
    echo "Submitted Chunk (${start}-${end} / ${TOTAL_TASKS}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "Tasks ${START_TASK} to ${END_TASK} successfully scheduled on cluster!"
echo "=================================================="
