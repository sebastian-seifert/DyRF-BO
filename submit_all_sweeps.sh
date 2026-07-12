#!/bin/bash

# Submit the 738-run unified UQ sweep to the LUIS cluster in 3 sequential chunks
# to comply with the MaxArraySize limit of 300 tasks per array job,
# while maintaining a strict resource allocation ceiling of 8 physical A100 GPUs (24 concurrent tasks).

echo "Submitting unified sweep to cluster..."

# Submit Chunk 1 (Tasks 1-250)
JOB1=$(sbatch --parsable --array=1-250%24 run_unified_cluster_sweep.sh)
echo "Submitted chunk 1 (Tasks 1-250) -> Job ID: $JOB1"

# Submit Chunk 2 (Tasks 251-500) - Starts only after Chunk 1 completes
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 --array=251-500%24 run_unified_cluster_sweep.sh)
echo "Submitted chunk 2 (Tasks 251-500) -> Job ID: $JOB2 (Dependent on $JOB1)"

# Submit Chunk 3 (Tasks 501-738) - Starts only after Chunk 2 completes
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 --array=501-738%24 run_unified_cluster_sweep.sh)
echo "Submitted chunk 3 (Tasks 501-738) -> Job ID: $JOB3 (Dependent on $JOB2)"

echo "--------------------------------------------------------"
echo "All sweep chunks successfully staged in SLURM queue."
echo "Total executions: 738 runs (Seeds=5, Linear scaling only)."
echo "Max concurrent allocation: 8 physical A100 GPUs (24 tasks)."
echo "--------------------------------------------------------"
