#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Submitting BBOB Dim 128 Sweep Part 1 (Tasks 1 to 3,600)"
echo "=================================================="

START_TASK=1 END_TASK=3600 ./scripts/submit_bbob_dim128_sweep.sh
