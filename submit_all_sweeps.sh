#!/bin/bash
# Master Sweep Launcher Script
# Triggers all 6 independent test sweeps sequentially onto the SLURM cluster
# with 1 single SLURM Array Job per sweep, dependency-chained (exactly 6 queued jobs total!).

set -e

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

echo "=========================================================="
echo "LAUNCHING MASTER UQ BENCHMARK SWEEP SUITE (SINGLE-ARRAY CHAINED)"
echo "=========================================================="

echo -e "\n[1/6] Launching Sweep 1: Empty Hypercube Gap..."
OUT1=$(./scripts/submit_sweep1_empty.sh)
echo "$OUT1"
JOB1=$(echo "$OUT1" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[2/6] Launching Sweep 2: Linear Sparse Hypercube Gap (chained after Job $JOB1)..."
OUT2=$(./scripts/submit_sweep2_linear_sparse.sh "$JOB1")
echo "$OUT2"
JOB2=$(echo "$OUT2" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[3/6] Launching Sweep 3: Fractional Sparse Hypercube Gap (chained after Job $JOB2)..."
OUT3=$(./scripts/submit_sweep3_fractional_sparse.sh "$JOB2")
echo "$OUT3"
JOB3=$(echo "$OUT3" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[4/6] Launching Sweep 4: Leaf Sparse Hypercube Gap (chained after Job $JOB3)..."
OUT4=$(./scripts/submit_sweep4_leaf_sparse.sh "$JOB3")
echo "$OUT4"
JOB4=$(echo "$OUT4" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[5/6] Launching Sweep 5: Manifold OOD Generation (chained after Job $JOB4)..."
OUT5=$(./scripts/submit_sweep5_manifold.sh "$JOB4")
echo "$OUT5"
JOB5=$(echo "$OUT5" | grep "LAST_JOB_ID:" | cut -d':' -f2)

echo -e "\n[6/6] Launching Sweep 6: CARP-S Epistemic EI Re-run (chained after Job $JOB5)..."
OUT6=$(./scripts/submit_sweep6_carps_epistemic_ei.sh "$JOB5")
echo "$OUT6"

echo -e "\n=========================================================="
echo "ALL 6 SWEEPS SUCCESSFULLY SCHEDULED ON SLURM CLUSTER!"
echo "Total SLURM Array Jobs Queued: EXACTLY 6 (1 job per sweep)"
echo "=========================================================="
echo "To monitor progress for any sweep, run:"
echo "  python3 scripts/monitor_progress.py results/sweep_1_empty"
echo "  python3 scripts/monitor_progress.py results/carps_epistemic_ei_scaled"
echo "=========================================================="
