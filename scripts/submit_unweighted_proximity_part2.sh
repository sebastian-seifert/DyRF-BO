#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Submitting Unweighted Proximity Sweep Part 2 (Tasks 5,001 to 9,600)"
echo "=================================================="

START_TASK=5001 END_TASK=9600 ./scripts/submit_unweighted_proximity_sweep.sh
