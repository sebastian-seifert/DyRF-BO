#!/bin/bash
# Submit the 130 Proximity Auto Lambda CARP-S tasks (26 tasks x 5 seeds).

mkdir -p results

if [ ! -f "results/array_tasks_dynamic_lambda.txt" ]; then
    echo "results/array_tasks_dynamic_lambda.txt not found. Generating array tasks..."
    python3 scripts/generate_dynamic_lambda_array_tasks.py
fi

JOB=$(sbatch --parsable --array=1-130%15 scripts/submit_dynamic_lambda_array.sbatch)
echo "Submitted Proximity Auto Lambda Array (Tasks 1-130) -> Job ID: $JOB"

echo "--------------------------------------------------------"
echo "All 130 Proximity Auto Lambda sweep runs successfully scheduled on LUIS cluster."
echo "Max concurrent allocation: 15 tasks."
echo "--------------------------------------------------------"
