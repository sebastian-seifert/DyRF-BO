#!/bin/bash
set -e

# Detect virtual environment python
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
else
    PYTHON_EXEC="python"
fi

# Configuration details
TASK="+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_3"
TRIALS=15
SEED=1

# List of all 7 approaches
APPROACHES=(
    "standard_disagreement"
    "chen_variance"
    "shaker_entropy"
    "likelihood_credal"
    "standard_proximity"
    "proximity_b"
    "proximity_bc"
)

echo "=========================================================="
echo "STARTING INTERACTIVE SANITY CHECKS FOR ALL 7 APPROACHES"
echo "Task: $TASK"
echo "Trials: $TRIALS | Seed: $SEED"
echo "=========================================================="

export PYTHONPATH=.

for extractor in "${APPROACHES[@]}"; do
    echo ""
    echo "--------------------------------------------------------"
    echo "Running Approach: $extractor"
    echo "--------------------------------------------------------"
    
    $PYTHON_EXEC scripts/run_carps_patched.py \
        --config-dir carps_integration/configs \
        +optimizer=dyrf_epistemic_hpobench \
        optimizer.extractor_name="$extractor" \
        $TASK \
        task.optimization_resources.n_trials=$TRIALS \
        seed=$SEED
        
    echo "✓ Finished $extractor successfully!"
done

echo ""
echo "=========================================================="
echo "ALL 7 SANITY CHECK RUNS COMPLETED SUCCESSFULLY!"
echo "=========================================================="
