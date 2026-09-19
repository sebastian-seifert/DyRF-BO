#!/bin/bash
# Master submission script for yahpo_rbv2_ranger
# Total Tasks: 7140 across 3 chunked array jobs
set -e

echo 'Submitting submit_sweep_yahpo_rbv2_ranger_proximity_p1.sbatch...'
sbatch scripts/submit_sweep_yahpo_rbv2_ranger_proximity_p1.sbatch

echo 'Submitting submit_sweep_yahpo_rbv2_ranger_proximity_p2.sbatch...'
sbatch scripts/submit_sweep_yahpo_rbv2_ranger_proximity_p2.sbatch

echo 'Submitting submit_sweep_yahpo_rbv2_ranger_proximity_p3.sbatch...'
sbatch scripts/submit_sweep_yahpo_rbv2_ranger_proximity_p3.sbatch

