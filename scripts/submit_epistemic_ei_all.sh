#!/bin/bash
# Submit the 1,170 streamlined UQ sweep tasks in 4 sequential dependent chunks
# to comply with LUIS cluster MaxArraySize=300 limit.

mkdir -p results

# Auto-generate results/epistemic_ei_array_tasks.txt if missing
if [ ! -f "results/epistemic_ei_array_tasks.txt" ]; then
    echo "results/epistemic_ei_array_tasks.txt not found. Generating array tasks..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_epistemic_ei_array_tasks.py
    else
        python3 scripts/generate_epistemic_ei_array_tasks.py
    fi
fi

# Chunk 1 (1-300)
JOB1=$(sbatch --parsable --array=1-300%15 scripts/submit_epistemic_ei_array.sbatch)
echo "Submitted Chunk 1 (Tasks 1-300) -> Job ID: $JOB1"

# Chunk 2 (301-600) - starts only after Chunk 1 finishes
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 --array=301-600%15 scripts/submit_epistemic_ei_array.sbatch)
echo "Submitted Chunk 2 (Tasks 301-600) -> Job ID: $JOB2 (dependent on $JOB1)"

# Chunk 3 (601-900) - starts only after Chunk 2 finishes
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 --array=601-900%15 scripts/submit_epistemic_ei_array.sbatch)
echo "Submitted Chunk 3 (Tasks 601-900) -> Job ID: $JOB3 (dependent on $JOB2)"

# Chunk 4 (901-1170) - starts only after Chunk 3 finishes
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 --array=901-1170%15 scripts/submit_epistemic_ei_array.sbatch)
echo "Submitted Chunk 4 (Tasks 901-1170) -> Job ID: $JOB4 (dependent on $JOB3)"

echo "--------------------------------------------------------"
echo "All 1,170 sweep runs successfully scheduled on LUIS cluster."
echo "Max concurrent allocation per array: 15 tasks."
echo "--------------------------------------------------------"
