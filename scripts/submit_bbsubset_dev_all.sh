#!/bin/bash
set -e

# Ensure output directories exist
mkdir -p results/bbsubset_dev/ei
mkdir -p results/bbsubset_dev/pi
mkdir -p results/bbsubset_dev/lcb
mkdir -p results/bbsubset_dev/baseline/ei
mkdir -p results/bbsubset_dev/baseline/pi
mkdir -p results/bbsubset_dev/baseline/lcb
mkdir -p results/bbsubset_dev/logs

# Ensure task list is generated
if [ ! -f "results/bbsubset_dev_tasks.txt" ]; then
    echo "Generating CARP-S BBsubset dev array tasks list..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_bbsubset_dev_tasks.py
    else
        python3 scripts/generate_bbsubset_dev_tasks.py
    fi
fi

TOTAL_TASKS=$(wc -l < results/bbsubset_dev_tasks.txt)
CHUNK_SIZE=250
echo "Submitting ${TOTAL_TASKS} tasks for CARP-S BBsubset Dev Sweep in chunks of ${CHUNK_SIZE} (MaxArraySize compliant)..."

for (( start=1; start<=TOTAL_TASKS; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $TOTAL_TASKS ]; then
        end=$TOTAL_TASKS
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%25 scripts/submit_bbsubset_dev_array.sbatch)
    echo "Submitted Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
done

echo "--------------------------------------------------------"
echo "CARP-S BBsubset Dev Sweep successfully scheduled on SLURM!"
echo "--------------------------------------------------------"
