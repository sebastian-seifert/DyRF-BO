#!/bin/bash
# Exit immediately if any test fails
set -e

# Detect local virtual environment or default python
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
    echo "Using local virtual environment: $PYTHON_EXEC"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
    echo "Using system python3"
else
    PYTHON_EXEC="python"
    echo "Using default python"
fi

echo "=================================================="
echo "RUNNING ALL UQ BENCHMARK TESTS"
echo "=================================================="

echo ""
echo ">> [1/2] Running Vectorized GPU/CPU Parity Test..."
$PYTHON_EXEC tests/verify_gpu_proximity.py

echo ""
echo ">> [2/3] Running 1D End-to-End Smoke Test..."
$PYTHON_EXEC tests/smoke_test.py

echo ""
echo ">> [3/3] Running All Unit Tests (Parallel Execution)..."
$PYTHON_EXEC scripts/run_tests_parallel.py

echo ""
echo "=================================================="
echo "ALL TESTS PASSED SUCCESSFULLY!"
echo "=================================================="
