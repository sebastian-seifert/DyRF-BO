#!/bin/bash
# Master Sweep Launcher Script
# Triggers all 6 independent test sweeps sequentially onto the SLURM cluster
# with inter-sweep job dependency chaining to prevent QOSMaxSubmitJobPerUserLimit errors.

set -e

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

echo "=========================================================="
echo "LAUNCHING MASTER UQ BENCHMARK SWEEP SUITE (DEPENDENCY CHAINED)"
echo "=========================================================="

echo -e "\n[1/6] Launching Sweep 1: Empty Hypercube Gap..."
OUT1=$(./scripts/submit_sweep1_empty.sh)
echo "$OUT1"
LAST_JOB1=$(echo "$OUT1" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[2/6] Launching Sweep 2: Linear Sparse Hypercube Gap (chained after Sweep 1 job $LAST_JOB1)..."
OUT2=$(./scripts/submit_sweep2_linear_sparse.sh "$LAST_JOB1")
echo "$OUT2"
LAST_JOB2=$(echo "$OUT2" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[3/6] Launching Sweep 3: Fractional Sparse Hypercube Gap (chained after Sweep 2 job $LAST_JOB2)..."
OUT3=$(./scripts/submit_sweep3_fractional_sparse.sh "$LAST_JOB2")
echo "$OUT3"
LAST_JOB3=$(echo "$OUT3" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[4/6] Launching Sweep 4: Leaf Sparse Hypercube Gap (chained after Sweep 3 job $LAST_JOB3)..."
OUT4=$(./scripts/submit_sweep4_leaf_sparse.sh "$LAST_JOB3")
echo "$OUT4"
LAST_JOB4=$(echo "$OUT4" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[5/6] Launching Sweep 5: Manifold OOD Generation (chained after Sweep 4 job $LAST_JOB4)..."
OUT5=$(./scripts/submit_sweep5_manifold.sh "$LAST_JOB4")
echo "$OUT5"
LAST_JOB5=$(echo "$OUT5" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[6/6] Launching Sweep 6: CARP-S Epistemic EI Re-run (chained after Sweep 5 job $LAST_JOB5)..."
OUT6=$(./scripts/submit_sweep6_carps_epistemic_ei.sh "$LAST_JOB5")
echo "$OUT6"

echo -e "\n=========================================================="
echo "ALL 6 SWEEPS SUCCESSFULLY SCHEDULED ON SLURM CLUSTER!"
echo "Inter-sweep dependencies active: Each sweep starts after the previous finishes."
echo "=========================================================="
echo "To monitor progress for any sweep, run:"
echo "  python3 scripts/monitor_progress.py results/sweep_1_empty"
echo "  python3 scripts/monitor_progress.py results/carps_epistemic_ei_scaled"
echo "=========================================================="
