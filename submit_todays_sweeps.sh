#!/bin/bash
# Dedicated Today's Sweeps Launcher Script
# Submits Sweep 1, Sweep 2, and the CARP-S Epistemic EI Re-run cleanly in parallel.

set -e

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

echo "=========================================================="
echo "LAUNCHING TODAY'S SWEEPS (SWEEP 1, SWEEP 2, CARP-S EPISTEMIC EI)"
echo "=========================================================="

echo -e "\n[1/3] Launching Sweep 1: Empty Hypercube Gap..."
./scripts/submit_sweep1_empty.sh

echo -e "\n[2/3] Launching Sweep 2: Linear Sparse Hypercube Gap..."
./scripts/submit_sweep2_linear_sparse.sh

echo -e "\n[3/3] Launching Sweep 3 (CARP-S): EU-Guided EI CARP-S Re-run (Scaled Signals)..."
./scripts/submit_sweep6_carps_epistemic_ei.sh

echo -e "\n=========================================================="
echo "TODAY'S SWEEPS SUCCESSFULLY SCHEDULED ON SLURM CLUSTER!"
echo "=========================================================="
echo "To monitor progress for any sweep, run:"
echo "  python3 scripts/monitor_progress.py results/sweep_1_empty"
echo "  python3 scripts/monitor_progress.py results/sweep_2_linear_sparse"
echo "  python3 scripts/monitor_progress.py results/carps_epistemic_ei_scaled"
echo "=========================================================="
