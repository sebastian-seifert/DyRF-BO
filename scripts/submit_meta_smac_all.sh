#!/bin/bash
set -e

echo "=================================================="
echo "SMAC4HPO Meta-Optimization on Proximity LCB"
echo "Running 100 Bayesian Optimization Iterations on CARP-S BBsubset Dev Set"
echo "(18 working dev tasks * 5 seeds = 90 runs per iteration, well below 5,000 limit)"
echo "=================================================="

# Ensure reference bounds exist
if [ ! -f "results/meta_smac_proximity_hpo/reference_bounds.json" ]; then
    echo "Extracting empirical reference bounds from CARP-S logs..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/extract_dev_reference_bounds.py
    else
        python3 scripts/extract_dev_reference_bounds.py
    fi
fi

if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "Launching SMAC4HPO Orchestrator for 100 Iterations..."

$PYTHON_BIN scripts/run_meta_smac_proximity_hpo.py \
    --start-iteration 1 \
    --end-iteration 100 \
    --seeds 5 \
    --trials 100 \
    --output-dir results/meta_smac_proximity_hpo \
    --baserundir runs/meta_smac_proximity_hpo \
    "$@"

echo "=================================================="
echo "SMAC4HPO Meta-Optimization Completed Successfully!"
echo "Final Best Configuration: results/meta_smac_proximity_hpo/best_config.json"
echo "Full Iteration Leaderboard: results/meta_smac_proximity_hpo/meta_leaderboard.csv"
echo "=================================================="
