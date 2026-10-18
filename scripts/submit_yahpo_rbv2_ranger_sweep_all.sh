#!/bin/bash
set -e

echo "=================================================="
echo "Preparing CARP-S YAHPO rbv2_ranger Master Scheduler"
echo "=================================================="

RESULTS_DIR="results/sweep_yahpo_rbv2_ranger"
TASK_FILE="${RESULTS_DIR}/tasks.txt"
CHUNK_SIZE=250

if [ ! -f "$TASK_FILE" ]; then
    echo "Generating task manifest for YAHPO rbv2_ranger..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_yahpo_rbv2_ranger_sweep_tasks.py
    else
        python3 scripts/generate_yahpo_rbv2_ranger_sweep_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
echo "Total YAHPO rbv2_ranger Tasks: ${TOTAL_TASKS}"
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
    echo "Submitting Master Scheduler to Slurm via scripts/submit_yahpo_rbv2_ranger_sweep_master.sbatch..."
    JOB_ID=$(sbatch --parsable scripts/submit_yahpo_rbv2_ranger_sweep_master.sbatch "${PASSTHROUGH_ARGS[@]}")
    echo "Submitted Master Orchestrator Job: ${JOB_ID}"
    echo "Monitor progress with: tail -f ${RESULTS_DIR}/logs/master_orchestrator_${JOB_ID}.out"
else
    echo "Starting Master Slurm Scheduler locally (chunk size: ${CHUNK_SIZE}, array concurrency: %25)..."
    STATE_FILE_ARG=()
    HAS_CUSTOM_STATE=false
    IS_DRY_RUN=false
    for arg in "${PASSTHROUGH_ARGS[@]}"; do
        if [ "$arg" == "--state-file" ]; then
            HAS_CUSTOM_STATE=true
        fi
        if [ "$arg" == "--dry-run" ]; then
            IS_DRY_RUN=true
        fi
    done

    if [ "$HAS_CUSTOM_STATE" = false ]; then
        if [ "$IS_DRY_RUN" = true ]; then
            STATE_FILE_ARG=(--state-file "${RESULTS_DIR}/scheduler_state.dry_run.json")
        else
            STATE_FILE_ARG=(--state-file "${RESULTS_DIR}/scheduler_state.json")
        fi
    fi

    exec "$PYTHON_BIN" scripts/master_slurm_scheduler.py \
        --manifest "$TASK_FILE" \
        --sbatch scripts/submit_yahpo_rbv2_ranger_sweep_array.sbatch \
        --chunk-size "$CHUNK_SIZE" \
        --max-tasks 4500 \
        --trigger-jobs 10 \
        --concurrency 25 \
        "${STATE_FILE_ARG[@]}" \
        --job-prefix rngr \
        "${PASSTHROUGH_ARGS[@]}"
fi
