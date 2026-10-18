#!/usr/bin/env bash
# ==============================================================================
# Master Execution Script for All YAHPO Analysis Tasks (CARP-S Reports + Custom Scorecards)
# ==============================================================================
# Usage:
#   ./scripts/run_all_yahpo_analyses.sh --dry-run
#   ./scripts/run_all_yahpo_analyses.sh --tasks ranger_all
#   ./scripts/run_all_yahpo_analyses.sh --tasks super_all
#   ./scripts/run_all_yahpo_analyses.sh --tasks carps_all
#   ./scripts/run_all_yahpo_analyses.sh --tasks custom_all
#   ./scripts/run_all_yahpo_analyses.sh --tasks all
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# Activate virtualenv if present
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
elif [[ -f "/bigwork/nhwpseis/.conda/envs/dyrf/bin/activate" ]]; then
    source "/bigwork/nhwpseis/.conda/envs/dyrf/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "[START] Running YAHPO Unified Analysis Pipeline: python -m scripts.run_all_yahpo_analyses $@"
python3 scripts/run_all_yahpo_analyses.py "$@"
echo "[DONE] YAHPO Unified Analysis Pipeline finished successfully."
