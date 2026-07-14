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

# Run CARP-S with our custom patched launcher
$PYTHON_EXEC scripts/run_carps_patched.py \
    --config-dir carps_integration/configs \
    "$@"

echo "=================================================="
echo "BENCHMARK SWEEP STEP COMPLETED SUCCESSFULLY!"
echo "=================================================="
