#!/usr/bin/env bash
# ==============================================================================
# HPOBench Verification Script Wrapper
# Resolves cluster conda or local venv environment and executes verification
# ==============================================================================

set -e

# Resolve cluster conda environment or local venv
if [ -f "/bigwork/nhwpseis/.conda/envs/dyrf/bin/activate" ]; then
    source "/bigwork/nhwpseis/.conda/envs/dyrf/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
elif [ -f "../.venv/bin/activate" ]; then
    source "../.venv/bin/activate"
fi

# Set default cluster environment variables if not already set
if [ -z "$CARPS_TASK_DATA_DIR" ] && [ -d "/bigwork/nhwpseis/benchmarks" ]; then
    export CARPS_TASK_DATA_DIR="/bigwork/nhwpseis/benchmarks"
fi

if [ -z "$HPOBENCH_DATA_DIR" ] && [ -d "/bigwork/nhwpseis/benchmarks/hpobench" ]; then
    export HPOBENCH_DATA_DIR="/bigwork/nhwpseis/benchmarks/hpobench"
fi

# Execute python verification script passing all CLI arguments
exec python3 scripts/verify_hpobench_tasks_available.py "$@"
