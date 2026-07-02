#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Define parameters for the experimental grid (moderately downscaled for efficiency)
RF_CONFIGS=(1 3 5)
K_VALUES=(20 100 500)
SPARSE_MULTIPLIERS=(5 15 50)
SCALING_LAWS=("linear" "leaf")

# Default: run both gap types if none specified
GAP_TYPES=("empty" "sparse")

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gap_type) GAP_TYPES=("$2"); shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Calculate total runs
total=0
for gap_type in "${GAP_TYPES[@]}"; do
    if [ "$gap_type" == "sparse" ]; then
        total=$(( total + 25 * ${#SPARSE_MULTIPLIERS[@]} * ${#SCALING_LAWS[@]} ))
    else
        total=$(( total + 25 ))
    fi
done

echo "=========================================================="
echo "LAUNCHING EPISTEMIC UQ PROXIMITY GRID SEARCH BENCHMARKS"
echo "Grid size: 5 RF Configs x 5 K Values x ${#GAP_TYPES[@]} Gap Types = $total runs (10 seeds each)"
echo "Gap Types: ${GAP_TYPES[*]}"
echo "Sparse Multipliers: ${SPARSE_MULTIPLIERS[*]}"
echo "Scaling Laws: ${SCALING_LAWS[*]}"
echo "Estimated total runtime: ~6-8 hours on CPU, ~45-60 minutes on GPU (e.g. A100)"
echo "=========================================================="

# Detect active Python environment or local venv
if [ -f "../.venv/bin/python" ]; then
    PYTHON_EXEC="../.venv/bin/python"
    echo "Using local virtual environment: $PYTHON_EXEC"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
    echo "Using system python3"
else
    PYTHON_EXEC="python"
    echo "Using default python"
fi

current=1
start_time=$(date +%s)

for gap_type in "${GAP_TYPES[@]}"; do
    for config in "${RF_CONFIGS[@]}"; do
        for k in "${K_VALUES[@]}"; do
            if [ "$gap_type" == "sparse" ]; then
                for law in "${SCALING_LAWS[@]}"; do
                    for mult in "${SPARSE_MULTIPLIERS[@]}"; do
                        run_start=$(date +%s)
                        percent=$(( (current - 1) * 100 / total ))
                        echo ""
                        echo "----------------------------------------------------------"
                        echo "[Run $current/$total] ($percent% Complete) Config: RF=$config, K=$k, Gap=sparse, Law=$law, Multiplier=$mult"
                        echo "Started: $(date)"
                        echo "----------------------------------------------------------"
                        
                        $PYTHON_EXEC Uncertainty_Quantification.py --rf_config "$config" --k_neighbors "$k" --gap_type "sparse" --sparse_multiplier "$mult" --scaling_law "$law" --n_runs 5
                        
                        run_end=$(date +%s)
                        duration=$((run_end - run_start))
                        echo "Finished [Run $current/$total] in $((duration / 60))m $((duration % 60))s"
                        current=$((current + 1))
                    done
                done
            else
                # gap_type == "empty"
                run_start=$(date +%s)
                percent=$(( (current - 1) * 100 / total ))
                echo ""
                echo "----------------------------------------------------------"
                echo "[Run $current/$total] ($percent% Complete) Config: RF=$config, K=$k, Gap=empty"
                echo "Started: $(date)"
                echo "----------------------------------------------------------"
                
                $PYTHON_EXEC Uncertainty_Quantification.py --rf_config "$config" --k_neighbors "$k" --gap_type "empty" --n_runs 5
                
                run_end=$(date +%s)
                duration=$((run_end - run_start))
                echo "Finished [Run $current/$total] in $((duration / 60))m $((duration % 60))s"
                current=$((current + 1))
            fi
        done
    done
done

end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo ""
echo "=========================================================="
echo "ALL BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Total runtime: $((total_duration / 3600))h $(((total_duration % 3600) / 60))m $((total_duration % 60))s"
echo "=========================================================="
