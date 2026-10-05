#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Submitting Lower-Quantile UQ Sweep Part 1 (Tasks 1 to 5,000)"
echo "=================================================="

START_TASK=1 END_TASK=5000 ./scripts/submit_lower_quantile_sweep.sh
