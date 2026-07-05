#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Define parameters for the variants sweep
RF_CONFIGS=(1 3 5)
GAP_TYPES=("empty" "sparse")

# Non-proximity approaches to evaluate
APPROACHES="Standard,Chen,Shaker,Credal_GL_Bisect,Credal_GL_Newton,Credal_Trapz_Bisect,Credal_Trapz_Newton"

# Allocating CPU cores per job to share 32 cores: 6 jobs x 5 cores = 30 cores total
CORES_PER_JOB=5

# Create results and logging directories
mkdir -p results/logs

echo "=========================================================="
echo "LAUNCHING NON-PROXIMITY VARIANTS SWEEP IN PARALLEL"
echo "Approaches:    $APPROACHES"
echo "RF Configs:    ${RF_CONFIGS[*]}"
echo "Gap Types:     ${GAP_TYPES[*]}"
echo "Allocating:    $CORES_PER_JOB CPU cores per job (saturating 30/32 cores total)"
echo "Logs saved to: results/logs/"
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

current=1
total_runs=$(( ${#RF_CONFIGS[@]} * ${#GAP_TYPES[@]} ))
start_time=$(date +%s)

for gap_type in "${GAP_TYPES[@]}"; do
    for config in "${RF_CONFIGS[@]}"; do
        job_id=$current
        log_file="results/logs/run_variants_${config}_${gap_type}.log"
        
        echo "[$job_id/$total_runs] Dispatching Config: RF=$config, Gap=$gap_type (Logging to $log_file)"
        
        if [ "$gap_type" == "sparse" ]; then
            ( $PYTHON_EXEC Uncertainty_Quantification.py \
                --rf_config "$config" \
                --gap_type "sparse" \
                --sparse_multiplier 12 \
                --scaling_law "linear" \
                --approaches "$APPROACHES" \
                --n_jobs "$CORES_PER_JOB" \
                --n_runs 10 > "$log_file" 2>&1 \
                && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" \
                || echo "   ✗ [Job $job_id/$total_runs] FAILED (check $log_file)" ) &
        else
            # gap_type == "empty"
            ( $PYTHON_EXEC Uncertainty_Quantification.py \
                --rf_config "$config" \
                --gap_type "empty" \
                --approaches "$APPROACHES" \
                --n_jobs "$CORES_PER_JOB" \
                --n_runs 10 > "$log_file" 2>&1 \
                && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" \
                || echo "   ✗ [Job $job_id/$total_runs] FAILED (check $log_file)" ) &
        fi
        
        current=$((current + 1))
    done
done

echo ""
echo "All 6 jobs dispatched to background. Waiting for processes to complete..."
wait

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================================="
echo "ALL NON-PROXIMITY VARIANTS COMPLETED SUCCESSFULLY!"
echo "Total runtime: $((duration / 60))m $((duration % 60))s"
echo "=========================================================="
