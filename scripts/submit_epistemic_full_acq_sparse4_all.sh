#!/bin/bash
# Submit the 540 EU-guided Multi-Acquisition (EI/PI/LCB) Sparse 4-Task array sweep tasks in sequential dependent chunks
# of 300 tasks to comply with the LUIS cluster MaxArraySize=300 limit.

mkdir -p results/epistemic_full_acq_sparse4/logs results/epistemic_full_acq_sparse4/ei results/epistemic_full_acq_sparse4/pi results/epistemic_full_acq_sparse4/lcb results/epistemic_full_acq_sparse4/baseline

# Auto-generate results/epistemic_full_acq_sparse4_array_tasks.txt if missing
if [ ! -f "results/epistemic_full_acq_sparse4_array_tasks.txt" ]; then
    echo "results/epistemic_full_acq_sparse4_array_tasks.txt not found. Generating array tasks..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_epistemic_full_acq_sparse4_array_tasks.py
    else
        python3 scripts/generate_epistemic_full_acq_sparse4_array_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < results/epistemic_full_acq_sparse4_array_tasks.txt)
CHUNK_SIZE=300
PREV_JOB=""

echo "Submitting $TOTAL_TASKS Multi-Acquisition Sparse 4-Task tasks in chunks of $CHUNK_SIZE..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi

    if [ -z "$PREV_JOB" ]; then
        JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_epistemic_full_acq_sparse4_array.sbatch)
        echo "Submitted Tasks ${start}-${end} -> Job ID: ${JOB_ID}"
    else
        JOB_ID=$(sbatch --parsable --dependency=afterany:${PREV_JOB} --array=${start}-${end}%25 scripts/submit_epistemic_full_acq_sparse4_array.sbatch)
        echo "Submitted Tasks ${start}-${end} -> Job ID: ${JOB_ID} (dependent on ${PREV_JOB})"
    fi
    PREV_JOB=${JOB_ID}
done

echo "--------------------------------------------------------"
echo "All $TOTAL_TASKS Multi-Acquisition Sparse 4-Task sweep runs successfully scheduled on LUIS cluster."
echo "Max concurrent allocation per array: 25 tasks."
echo "--------------------------------------------------------"
