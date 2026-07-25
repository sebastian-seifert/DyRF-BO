#!/bin/bash
# Master Sweep Launcher Script
# Triggers all 6 independent test sweeps sequentially onto the SLURM cluster.

set -e

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

echo "=========================================================="
echo "LAUNCHING MASTER UQ BENCHMARK SWEEP SUITE"
echo "=========================================================="

echo -e "\n[1/6] Launching Sweep 1: Empty Hypercube Gap..."
./scripts/submit_sweep1_empty.sh

echo -e "\n[2/6] Launching Sweep 2: Linear Sparse Hypercube Gap..."
./scripts/submit_sweep2_linear_sparse.sh

echo -e "\n[3/6] Launching Sweep 3: Fractional Sparse Hypercube Gap..."
./scripts/submit_sweep3_fractional_sparse.sh

echo -e "\n[4/6] Launching Sweep 4: Leaf Sparse Hypercube Gap..."
./scripts/submit_sweep4_leaf_sparse.sh

echo -e "\n[5/6] Launching Sweep 5: Manifold OOD Generation..."
./scripts/submit_sweep5_manifold.sh

echo -e "\n[6/6] Launching Sweep 6: CARP-S Epistemic EI Re-run (Scaled Signals)..."
./scripts/submit_sweep6_carps_epistemic_ei.sh

echo -e "\n=========================================================="
echo "ALL 6 SWEEPS SUCCESSFULLY SCHEDULED ON SLURM CLUSTER!"
echo "=========================================================="
echo "To monitor progress for any sweep, run:"
echo "  python3 scripts/monitor_progress.py results/sweep_1_empty"
echo "  python3 scripts/monitor_progress.py results/carps_epistemic_ei_scaled"
echo "=========================================================="
