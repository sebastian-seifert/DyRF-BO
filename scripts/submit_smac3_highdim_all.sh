#!/bin/bash
# Submit standard SMAC3 BO baseline High-Dimensional array sweep tasks (90 tasks) to LUIS cluster

mkdir -p results/epistemic_ei_highdim/logs results/epistemic_ei_highdim/baseline

# Auto-generate results/smac3_highdim_array_tasks.txt if missing
if [ ! -f "results/smac3_highdim_array_tasks.txt" ]; then
    echo "results/smac3_highdim_array_tasks.txt not found. Generating array tasks..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_smac3_highdim_array_tasks.py
    else
        python3 scripts/generate_smac3_highdim_array_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < results/smac3_highdim_array_tasks.txt)

echo "Submitting $TOTAL_TASKS High-Dimensional SMAC3 baseline tasks..."

JOB_ID=$(sbatch --parsable --array=1-${TOTAL_TASKS}%15 scripts/submit_smac3_highdim_array.sbatch)

echo "--------------------------------------------------------"
echo "Submitted SMAC3 High-Dim Baseline Array Job ID: ${JOB_ID}"
echo "Total tasks scheduled: ${TOTAL_TASKS}"
echo "Max concurrent allocation per array: 15 tasks."
echo "--------------------------------------------------------"
