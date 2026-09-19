#!/bin/bash
set -e

TASK_FILE="results/sweep_yahpo_rbv2_ranger_proximity/tasks.txt"
SBATCH_FILE="scripts/submit_sweep_yahpo_rbv2_ranger_proximity_array.sbatch"

if [ ! -f "$TASK_FILE" ] || [ ! -s "$TASK_FILE" ]; then
    echo "ERROR: $TASK_FILE does not exist or is empty." >&2
    exit 1
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE" | tr -d ' ')
CHUNK_SIZE=200
CONCURRENCY=25

echo "=================================================="
echo "Submitting $TOTAL_TASKS tasks for yahpo_rbv2_ranger"
echo "Chunk Size: $CHUNK_SIZE (LUIS MaxArraySize <= 300, %$CONCURRENCY concurrency)"
echo "=================================================="

# Optional range overrides from CLI: e.g. ./script.sh [START_TASK] [END_TASK]
REQ_START=${1:-1}
REQ_END=${2:-$TOTAL_TASKS}

if [ "$REQ_START" -lt 1 ]; then REQ_START=1; fi
if [ "$REQ_END" -gt "$TOTAL_TASKS" ]; then REQ_END=$TOTAL_TASKS; fi

for (( start=REQ_START; start<=REQ_END; start+=CHUNK_SIZE )); do
    end=$(( start + CHUNK_SIZE - 1 ))
    if [ $end -gt $REQ_END ]; then
        end=$REQ_END
    fi
    JOB_ID=$(sbatch --parsable --array=${start}-${end}%${CONCURRENCY} "$SBATCH_FILE")
    echo "Submitted Chunk (${start}-${end} / ${TOTAL_TASKS}) -> Job ID: ${JOB_ID}"
done

echo "=================================================="
echo "yahpo_rbv2_ranger successfully scheduled on LUIS cluster!"
echo "=================================================="
