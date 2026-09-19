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
        if [ -n "$SLURM_JOB_ID" ]; then
            PENDING_OR_RUNNING=$(squeue -u "$USER" -h -t R,PD | grep -v "^ *$SLURM_JOB_ID " | wc -l)
        else
            PENDING_OR_RUNNING=$(squeue -u "$USER" -h -t R,PD | wc -l)
        fi
        if [ "$PENDING_OR_RUNNING" -le 1 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queue is clear ($PENDING_OR_RUNNING active job(s) remaining, proceeding to next batch)."
            break
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Active jobs in queue: $PENDING_OR_RUNNING. Checking again in 60s..."
        sleep 60
    done
}

# Allow starting from a specific batch (default: 2, skipping already completed batch 1)
START_BATCH="${1:-2}"

echo "=================================================="
echo "Starting Automated Multi-Suite Sweep Orchestrator"
echo "Total Batches available: ${#BATCHES[@]}"
echo "Starting from Batch: $START_BATCH"
echo "=================================================="

for idx in "${!BATCHES[@]}"; do
    batch_script="${BATCHES[$idx]}"
    batch_num=$(( idx + 1 ))
    if [ "$batch_num" -lt "$START_BATCH" ]; then
        echo "Skipping Batch $batch_num (already completed): $batch_script"
        continue
    fi
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
