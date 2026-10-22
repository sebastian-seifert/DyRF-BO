#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=================================================="
echo "Submitting BBOB Dim 128 Sweep Part 2 (Tasks 3,601 to 7,200)"
echo "=================================================="

START_TASK=3601 END_TASK=7200 ./scripts/submit_bbob_dim128_sweep.sh
