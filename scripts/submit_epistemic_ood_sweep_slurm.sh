#!/bin/bash
# ==============================================================================
# Master Slurm Submission Wrapper for Epistemic OOD Sweep on LUIS Cluster
# ------------------------------------------------------------------------------
# LUIS Documentation Reference (docs.cluster.uni-hannover.de/doku.php/guide/slurm_usage_guide):
#   "Note: the maximum number of jobs in a job array is limited to 300.
#    The index number must be smaller than 1 million."
#
# Because this sweep consists of 2,040 tasks, direct submission of a single
# array (--array=1-2040) is rejected by Slurm with:
#   "sbatch: error: Invalid job array specification"
#
# To comply with LUIS scheduler limits, tasks are dispatched in chunks
# of 200 tasks (array size <= 300, index < 1,000,000) via submit_epistemic_ood_sweep_all.sh.
#
# USAGE (Run on login node):
#   bash scripts/submit_epistemic_ood_sweep_slurm.sh
#   (or: bash scripts/submit_epistemic_ood_sweep_all.sh)
# ==============================================================================

set -e

if [ -n "$SLURM_JOB_ID" ]; then
    echo "ERROR: Do NOT submit this master script via 'sbatch'!" >&2
    echo "On LUIS, job arrays cannot exceed 300 tasks." >&2
    echo "Please execute this script directly on the cluster login node:" >&2
    echo "  bash scripts/submit_epistemic_ood_sweep_slurm.sh" >&2
    exit 1
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DIR}/submit_epistemic_ood_sweep_all.sh" "$@"
