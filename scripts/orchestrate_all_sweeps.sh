#!/bin/bash
# Master Multi-Suite Proximity LCB Sweep Orchestrator
# Automatically executes sweep batches sequentially, waiting for queue to empty between batches.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

BATCHES=(
    "scripts/submit_sweep_yahpo_rbv2_ranger_proximity_batch1.sh"
    "scripts/submit_sweep_yahpo_rbv2_ranger_proximity_batch2.sh"
    "scripts/submit_sweep_yahpo_rbv2_super_proximity_batch1.sh"
    "scripts/submit_sweep_yahpo_rbv2_super_proximity_batch2.sh"
    "scripts/submit_sweep_hpobench_ml_proximity_batch1.sh"
    "scripts/submit_sweep_hpobench_ml_proximity_batch2.sh"
)

wait_for_queue_empty() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring SLURM queue for user $USER..."
    while true; do
        PENDING_OR_RUNNING=$(squeue -u "$USER" -h -t R,PD | wc -l)
        if [ "$PENDING_OR_RUNNING" -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queue is clear (0 active jobs)."
            break
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Active jobs in queue: $PENDING_OR_RUNNING. Checking again in 60s..."
        sleep 60
    done
}

echo "=================================================="
echo "Starting Automated Multi-Suite Sweep Orchestrator"
echo "Total Batches to execute: ${#BATCHES[@]}"
echo "=================================================="

for idx in "${!BATCHES[@]}"; do
    batch_script="${BATCHES[$idx]}"
    batch_num=$(( idx + 1 ))
    echo ""
    echo "=================================================="
    echo "Executing Batch $batch_num / ${#BATCHES[@]}: $batch_script"
    echo "=================================================="
    bash "$batch_script"

    echo "Batch $batch_num submitted. Waiting 15s for SLURM scheduler to update..."
    sleep 15

    wait_for_queue_empty
    echo "Batch $batch_num completed!"
done

echo ""
echo "=================================================="
echo "ALL MULTI-SUITE SWEEPS COMPLETED SUCCESSFULLY!"
echo "=================================================="
