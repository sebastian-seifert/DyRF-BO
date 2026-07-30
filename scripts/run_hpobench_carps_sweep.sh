#!/bin/bash
# Exit immediately if any command fails
set -e

# Detect virtual environment python
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
else
    PYTHON_EXEC="python"
fi

# Print run info
echo "=================================================="
# Print dynamic BO benchmarking description
echo "Running CARP-S HPOBench Dynamic BO Benchmark Sweep"
echo "Using python: $PYTHON_EXEC"
echo "=================================================="

# Set PYTHONPATH to include the current directory so carps_integration can be imported
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ensure task arguments were provided
if [ $# -eq 0 ] || [ -z "$1" ]; then
    echo "ERROR: No task arguments provided to run_hpobench_carps_sweep.sh!" >&2
    exit 1
fi

# Run CARP-S with our custom patched launcher
$PYTHON_EXEC scripts/run_carps_patched.py \
    --config-dir carps_integration/configs \
    ++conda_env_name=carps_env \
    "$@"

echo "=================================================="
echo "BENCHMARK SWEEP STEP COMPLETED SUCCESSFULLY!"
echo "=================================================="
