#!/bin/bash
# Master submission script for yahpo_rbv2_super
# Total Tasks: 6180 across 3 chunked array jobs
set -e

echo 'Submitting submit_array_part1.sbatch...'
sbatch results/sweep_yahpo_rbv2_super_proximity/submit_array_part1.sbatch

echo 'Submitting submit_array_part2.sbatch...'
sbatch results/sweep_yahpo_rbv2_super_proximity/submit_array_part2.sbatch

echo 'Submitting submit_array_part3.sbatch...'
sbatch results/sweep_yahpo_rbv2_super_proximity/submit_array_part3.sbatch

