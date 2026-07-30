#!/bin/bash
set -e

# Ensure output directories exist
mkdir -p results/epistemic_ei_pi_lcb_all_dim/ei
mkdir -p results/epistemic_ei_pi_lcb_all_dim/pi
mkdir -p results/epistemic_ei_pi_lcb_all_dim/lcb
mkdir -p results/epistemic_ei_pi_lcb_all_dim/baseline/ei
mkdir -p results/epistemic_ei_pi_lcb_all_dim/baseline/pi
mkdir -p results/epistemic_ei_pi_lcb_all_dim/baseline/lcb
mkdir -p results/epistemic_ei_pi_lcb_all_dim/logs

# Ensure task list is generated
if [ ! -f "results/epistemic_full_acq_array_tasks.txt" ]; then
    echo "Generating full acquisition array tasks list..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_epistemic_full_acq_array_tasks.py
    else
        python3 scripts/generate_epistemic_full_acq_array_tasks.py
    fi
fi

CHUNK_SIZE=250
echo "Submitting ${TOTAL_TASKS} tasks for Full Acquisition Sweep in chunks of ${CHUNK_SIZE} (MaxArraySize compliant)..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_epistemic_full_acq_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "--------------------------------------------------------"
echo "Full Acquisition Sweep successfully scheduled on SLURM!"
echo "--------------------------------------------------------"
