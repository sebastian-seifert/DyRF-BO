#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Define parameters for the variants sweep
RF_CONFIGS=(1 3 5)
GAP_TYPES=("empty" "sparse")

# Non-proximity approaches to evaluate
APPROACHES="Standard,Chen,Shaker,Credal_GL_Bisect,Credal_GL_Newton,Credal_Trapz_Bisect,Credal_Trapz_Newton"

echo "=========================================================="
echo "LAUNCHING NON-PROXIMITY VARIANTS SWEEP"
echo "Approaches: $APPROACHES"
echo "RF Configs: ${RF_CONFIGS[*]}"
echo "Gap Types:  ${GAP_TYPES[*]}"
echo "=========================================================="

# Detect active Python environment
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_EXEC="../.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
else
    PYTHON_EXEC="python"
fi
echo "Using python: $PYTHON_EXEC"

# Create results folder
mkdir -p results

current=1
total_runs=$(( ${#RF_CONFIGS[@]} * ${#GAP_TYPES[@]} ))

for gap_type in "${GAP_TYPES[@]}"; do
    for config in "${RF_CONFIGS[@]}"; do
        echo ""
        echo "----------------------------------------------------------"
        echo "[Run $current/$total_runs] Config: RF=$config, Gap=$gap_type"
        echo "Started: $(date)"
        echo "----------------------------------------------------------"
        
        if [ "$gap_type" == "sparse" ]; then
            $PYTHON_EXEC Uncertainty_Quantification.py \
                --rf_config "$config" \
                --gap_type "sparse" \
                --sparse_multiplier 12 \
                --scaling_law "linear" \
                --approaches "$APPROACHES" \
                --n_runs 10
        else
            # gap_type == "empty"
            $PYTHON_EXEC Uncertainty_Quantification.py \
                --rf_config "$config" \
                --gap_type "empty" \
                --approaches "$APPROACHES" \
                --n_runs 10
        fi
        
        current=$((current + 1))
    done
done

echo ""
echo "=========================================================="
echo "ALL NON-PROXIMITY VARIANTS COMPLETED SUCCESSFULLY!"
echo "=========================================================="
