#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Preparing Extrapolation Uncertainty Quantification Sweep"
echo "=================================================="

RESULTS_DIR="results/extrapolation_uq"
TASK_FILE="${TASK_FILE:-results/extrapolation_sweep_tasks.txt}"
CHUNK_SIZE=200
CONCURRENCY=25
MAX_TASKS=4500
TRIGGER_JOBS=10
STATE_FILE="${RESULTS_DIR}/scheduler_state.json"

if [ ! -f "$TASK_FILE" ]; then
    echo "Generating task manifest for extrapolation sweep..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_extrapolation_sweep_tasks.py --no-metrics
    else
        python3 scripts/generate_extrapolation_sweep_tasks.py --no-metrics
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
echo "Total tasks in task file: $TOTAL_TASKS"

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: Task file is empty!" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}/logs"

if [ -f "/bigwork/nhwpseis/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="/bigwork/nhwpseis/.conda/envs/dyrf/bin/python"
elif [ -f "$HOME/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/dyrf/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="$(which python3 || which python)"
fi

# Check if user specifically requested launching via sbatch
SUBMIT_SBATCH=false
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [ "$arg" == "--sbatch" ]; then
        SUBMIT_SBATCH=true
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

if [ "$SUBMIT_SBATCH" = true ]; then
    echo "Submitting Master Scheduler to Slurm via scripts/submit_extrapolation_sweep_master.sbatch..."
    JOB_ID=$(sbatch --parsable scripts/submit_extrapolation_sweep_master.sbatch "${PASSTHROUGH_ARGS[@]}")
    echo "Submitted Master Orchestrator Job: ${JOB_ID}"
    echo "Monitor progress with: tail -f ${RESULTS_DIR}/logs/master_orchestrator_${JOB_ID}.out"
else
    echo "Starting Master Slurm Scheduler locally (chunk size: ${CHUNK_SIZE}, array concurrency: %${CONCURRENCY})..."
    exec "$PYTHON_BIN" scripts/master_slurm_scheduler.py \
        --manifest "$TASK_FILE" \
        --sbatch scripts/submit_extrapolation_sweep_array.sbatch \
        --chunk-size "$CHUNK_SIZE" \
        --max-tasks "$MAX_TASKS" \
        --trigger-jobs "$TRIGGER_JOBS" \
        --concurrency "$CONCURRENCY" \
        --state-file "$STATE_FILE" \
        --job-prefix extrap_uq \
        "${PASSTHROUGH_ARGS[@]}"
fi

