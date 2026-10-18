#!/usr/bin/env bash
# ==============================================================================
# HPOBench Tabular Data Extractor
# Ensures all 5 ML models (lr, nn, rf, svm, xgb) are cleanly unzipped on the
# login node to avoid multi-worker unzipping race conditions on compute nodes.
# ==============================================================================

set -euo pipefail

# 1. Resolve TabularData directory
DATA_DIR="${HPOBENCH_DATA_DIR:-${CARPS_TASK_DATA_DIR:-/bigwork/nhwpseis/benchmarks}/hpobench}/TabularData"

if [ ! -d "$DATA_DIR" ]; then
    FALLBACK="$HOME/.local/share/hpobench/TabularData"
    if [ -d "$FALLBACK" ]; then
        DATA_DIR="$FALLBACK"
    else
        echo "ERROR: TabularData directory not found at $DATA_DIR or $FALLBACK" >&2
        exit 1
    fi
fi

echo "=================================================="
echo "Extracting HPOBench Tabular Data in: $DATA_DIR"
echo "=================================================="

MODELS=("lr" "nn" "rf" "svm" "xgb")

for m in "${MODELS[@]}"; do
    ZIP_FILE="$DATA_DIR/$m/$m.zip"
    TARGET_DIR="$DATA_DIR/$m"
    if [ -f "$ZIP_FILE" ]; then
        echo -n "Checking $m... "
        unzip -n -q "$ZIP_FILE" -d "$TARGET_DIR"
        echo "✓ Ready!"
    else
        echo "Warning: $ZIP_FILE not found, skipping."
    fi
done

echo "=================================================="
echo "All available HPOBench Tabular models are extracted."
echo "=================================================="
