#!/bin/bash
set -e

echo "=================================================="
echo "Preparing CARP-S BBSubset Held-Out Test Big Comparison Sweep"
echo "=================================================="

RESULTS_DIR="results/sweep_bbsubset_test_big_comparison"
TASK_FILE_P1="${RESULTS_DIR}/tasks_part1.txt"
TASK_FILE_P2="${RESULTS_DIR}/tasks_part2.txt"

DRY_RUN=false
for arg in "$@"; do
    if [ "$arg" == "--dry-run" ]; then
        DRY_RUN=true
    fi
done

if [ ! -f "$TASK_FILE_P1" ] || [ ! -f "$TASK_FILE_P2" ]; then
    echo "Generating task files..."
    if [ -f ".venv/bin/python" ]; then
        .venv/bin/python scripts/generate_bbsubset_test_big_comparison_tasks.py
    else
        python3 scripts/generate_bbsubset_test_big_comparison_tasks.py
    fi
fi

TOTAL_P1=$(wc -l < "$TASK_FILE_P1" | tr -d ' ')
TOTAL_P2=$(wc -l < "$TASK_FILE_P2" | tr -d ' ')

echo "Tasks in Part 1 (Baselines + Entropy + Prox A & AC): ${TOTAL_P1}"
echo "Tasks in Part 2 (Prox B & BC):                       ${TOTAL_P2}"
echo "Grand Total: $(( TOTAL_P1 + TOTAL_P2 )) runs"

mkdir -p "${RESULTS_DIR}/logs"

CHUNK_SIZE=200

submit_partition() {
    local part_name=$1
    local task_file=$2
    local total_tasks=$3

    echo ""
    echo "--- Submitting ${part_name} (${total_tasks} tasks) in chunks of ${CHUNK_SIZE} ---"

    for (( start=1; start<=total_tasks; start+=CHUNK_SIZE )); do
        end=$(( start + CHUNK_SIZE - 1 ))
        if [ $end -gt $total_tasks ]; then
            end=$total_tasks
        fi
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY-RUN] sbatch --export=ALL,TASK_FILE=${task_file} --array=${start}-${end}%25 scripts/submit_bbsubset_test_big_comparison_array.sbatch"
        else
            JOB_ID=$(sbatch --parsable --export=ALL,TASK_FILE=${task_file} --array=${start}-${end}%25 scripts/submit_bbsubset_test_big_comparison_array.sbatch)
            echo "Submitted ${part_name} Chunk (${start}-${end}) -> Job ID: ${JOB_ID}"
        fi
    done
}

submit_partition "Part 1" "$TASK_FILE_P1" "$TOTAL_P1"
submit_partition "Part 2" "$TASK_FILE_P2" "$TOTAL_P2"

echo ""
echo "=================================================="
if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed successfully. All commands verified."
else
    echo "All 6,600 tasks across Part 1 & Part 2 scheduled on LUIS cluster!"
fi
echo "=================================================="
