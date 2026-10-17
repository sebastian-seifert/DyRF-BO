#!/usr/bin/env bash
# ==============================================================================
# Master Execution Script for All BBOB Analysis Tasks (CARP-S Reports + Custom Scorecards)
# ==============================================================================
# Usage:
#   ./scripts/run_all_bbob_analyses.sh --dry-run
#   ./scripts/run_all_bbob_analyses.sh --tasks dim2 dim4
#   ./scripts/run_all_bbob_analyses.sh --tasks carps_all
#   ./scripts/run_all_bbob_analyses.sh --tasks custom_all
#   ./scripts/run_all_bbob_analyses.sh --tasks all
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# Activate virtualenv if present
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "[START] Running BBOB Unified Analysis Pipeline: python -m scripts.run_all_bbob_analyses $@"
python3 scripts/run_all_bbob_analyses.py "$@"
echo "[DONE] BBOB Unified Analysis Pipeline finished successfully."
