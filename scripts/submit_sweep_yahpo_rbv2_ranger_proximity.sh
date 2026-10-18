#!/bin/bash
# ==============================================================================
# Legacy entry point redirected to Autonomous Master Slurm Scheduler
# Prevents queue flooding beyond LUIS cluster limits (max 5,000 tasks).
# ==============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Submitting YAHPO rbv2_ranger via Master Slurm Scheduler"
echo "Job limits: max 4,500 active tasks, chunk size 250 (%25 concurrency)"
echo "=================================================="

exec bash scripts/submit_yahpo_rbv2_ranger_sweep_all.sh "$@"

