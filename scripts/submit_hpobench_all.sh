#!/bin/bash
# Submit the 1040 HPOBench/YAHPO tasks in 4 dependent chunks of 260 tasks
# to comply with LUIS cluster MaxArraySize=300 limit.

mkdir -p results

# Auto-generate results/array_tasks.txt if missing
if [ ! -f "results/array_tasks.txt" ]; then
    echo "results/array_tasks.txt not found. Generating array tasks..."
    python3 scripts/generate_array_tasks.py
fi


# Chunk 1 (1-260)
JOB1=$(sbatch --parsable --array=1-260%15 scripts/submit_hpobench_array.sbatch)
echo "Submitted Chunk 1 (Tasks 1-260) -> Job ID: $JOB1"

# Chunk 2 (261-520) - starts only after Chunk 1 finishes
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 --array=261-520%15 scripts/submit_hpobench_array.sbatch)
echo "Submitted Chunk 2 (Tasks 261-520) -> Job ID: $JOB2"

# Chunk 3 (521-780) - starts only after Chunk 2 finishes
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 --array=521-780%15 scripts/submit_hpobench_array.sbatch)
echo "Submitted Chunk 3 (Tasks 521-780) -> Job ID: $JOB3"

# Chunk 4 (781-1040) - starts only after Chunk 3 finishes
JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 --array=781-1040%15 scripts/submit_hpobench_array.sbatch)
echo "Submitted Chunk 4 (Tasks 781-1040) -> Job ID: $JOB4"

echo "--------------------------------------------------------"
echo "All 1,040 sweep runs successfully scheduled on LUIS cluster."
echo "Max concurrent allocation per array: 15 tasks."
echo "--------------------------------------------------------"
