#!/bin/bash
# Submit the 3,510 multi-acquisition sweep tasks in sequential dependent chunks
# of 300 tasks to comply with the LUIS cluster MaxArraySize=300 limit.

mkdir -p results

# Auto-generate results/epistemic_acq_array_tasks.txt if missing
if [ ! -f "results/epistemic_acq_array_tasks.txt" ]; then
    echo "results/epistemic_acq_array_tasks.txt not found. Generating array tasks..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_epistemic_acq_array_tasks.py
    else
        python3 scripts/generate_epistemic_acq_array_tasks.py
    fi
fi

TOTAL_TASKS=3510
CHUNK_SIZE=300
PREV_JOB=""

echo "Submitting $TOTAL_TASKS tasks in chunks of $CHUNK_SIZE..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi

    if [ -z "$PREV_JOB" ]; then
        JOB_ID=$(sbatch --parsable --array=${start}-${end}%15 scripts/submit_epistemic_acq_array.sbatch)
        echo "Submitted Tasks ${start}-${end} -> Job ID: ${JOB_ID}"
    else
        JOB_ID=$(sbatch --parsable --dependency=afterany:${PREV_JOB} --array=${start}-${end}%15 scripts/submit_epistemic_acq_array.sbatch)
        echo "Submitted Tasks ${start}-${end} -> Job ID: ${JOB_ID} (dependent on ${PREV_JOB})"
    fi
    PREV_JOB=${JOB_ID}
done

echo "--------------------------------------------------------"
echo "All 3,510 multi-acquisition sweep runs successfully scheduled on LUIS cluster."
echo "Max concurrent allocation per array: 15 tasks."
echo "--------------------------------------------------------"
