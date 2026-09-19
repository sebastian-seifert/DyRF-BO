#!/bin/bash
# Master submission script for hpobench_ml
# Total Tasks: 5280 across 3 chunked array jobs
set -e

echo 'Submitting submit_array_part1.sbatch...'
sbatch results/sweep_hpobench_ml_proximity/submit_array_part1.sbatch

echo 'Submitting submit_array_part2.sbatch...'
sbatch results/sweep_hpobench_ml_proximity/submit_array_part2.sbatch

echo 'Submitting submit_array_part3.sbatch...'
sbatch results/sweep_hpobench_ml_proximity/submit_array_part3.sbatch

