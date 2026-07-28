#!/bin/bash
# Dedicated Today's Sweeps Launcher Script
# Submits Sweep 1 (Empty Hypercube Gap) and Sweep 2 (Linear Sparse Hypercube Gap) in parallel.

set -e

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

echo "=========================================================="
echo "LAUNCHING TODAY'S SWEEPS (SWEEP 1 & SWEEP 2)"
echo "=========================================================="

echo -e "\n[1/2] Launching Sweep 1: Empty Hypercube Gap..."
./scripts/submit_sweep1_empty.sh

echo -e "\n[2/2] Launching Sweep 2: Linear Sparse Hypercube Gap..."
./scripts/submit_sweep2_linear_sparse.sh

echo -e "\n=========================================================="
echo "TODAY'S SWEEPS (SWEEP 1 & 2) SUCCESSFULLY SCHEDULED ON SLURM CLUSTER!"
echo "=========================================================="
echo "To monitor progress for either sweep, run:"
echo "  python3 scripts/monitor_progress.py results/sweep_1_empty"
echo "  python3 scripts/monitor_progress.py results/sweep_2_linear_sparse"
echo "=========================================================="
