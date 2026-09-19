#!/bin/bash
# Master submission script for hpobench_ml
# Total Tasks: 5280 across 3 chunked array jobs
set -e

echo 'Submitting submit_sweep_hpobench_ml_proximity_p1.sbatch...'
sbatch scripts/submit_sweep_hpobench_ml_proximity_p1.sbatch

echo 'Submitting submit_sweep_hpobench_ml_proximity_p2.sbatch...'
sbatch scripts/submit_sweep_hpobench_ml_proximity_p2.sbatch

echo 'Submitting submit_sweep_hpobench_ml_proximity_p3.sbatch...'
sbatch scripts/submit_sweep_hpobench_ml_proximity_p3.sbatch

