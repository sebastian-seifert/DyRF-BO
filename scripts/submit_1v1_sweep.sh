#!/bin/bash
set -e

SWEEP_TYPE="${1:-all}"
CHUNK_SIZE=200

submit_single_sweep() {
    local name="$1"
    local task_file="results/sweep_1v1_${name}/tasks.txt"
    local sbatch_file="scripts/submit_1v1_${name}_array.sbatch"
    
    if [ ! -f "$task_file" ] || [ ! -s "$task_file" ]; then
        echo "Auto-generating task file for '${name}'..."
        if [ -f ".venv/bin/python" ]; then
            .venv/bin/python scripts/generate_1v1_sweep_tasks.py results 30
        else
            python3 scripts/generate_1v1_sweep_tasks.py results 30
        fi
    fi
    
    local total_tasks=$(wc -l < "$task_file" | tr -d ' ')
    echo "=================================================="
    echo "Submitting 1v1 Sweep: '${name}' (${total_tasks} tasks) in chunks of ${CHUNK_SIZE}..."
    echo "Task File:   ${task_file}"
    echo "SBatch File: ${sbatch_file}"
    echo "=================================================="
    
    for (( start=1; start<=total_tasks; start+=CHUNK_SIZE )); do
        end=$(( start + CHUNK_SIZE - 1 ))
        if [ $end -gt $total_tasks ]; then
            end=$total_tasks
        fi
        JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 "${sbatch_file}")
        echo "  -> Submitted Chunk (${start}-${end}) [${name}] | Job ID: ${JOB_ID}"
    done
    echo "[✓] Sweep '${name}' scheduled successfully!"
}

if [ "$SWEEP_TYPE" == "all" ]; then
    echo "Submitting ALL 3 1v1 Sweeps (30 seeds each, 3,600 total tasks)..."
    submit_single_sweep "disagreement"
    submit_single_sweep "proximity"
    submit_single_sweep "credal"
    echo ""
    echo "=================================================="
    echo "ALL 3 1v1 SWEEPS SCHEDULED ON SLURM CLUSTER!"
    echo "=================================================="
elif [ "$SWEEP_TYPE" == "disagreement" ] || [ "$SWEEP_TYPE" == "proximity" ] || [ "$SWEEP_TYPE" == "credal" ]; then
    submit_single_sweep "$SWEEP_TYPE"
else
    echo "Usage: $0 [all|disagreement|proximity|credal]"
    exit 1
fi
