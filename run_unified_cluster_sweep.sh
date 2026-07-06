#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Define parameters for the experimental grid
RF_CONFIGS=(1 3 5)
K_VALUES=(20 100 500)
SPARSE_MULTIPLIERS=(5 15 50)
SCALING_LAWS=("linear" "leaf")
GAP_TYPES=("empty" "sparse")
ALPHA_VALUES=(0.1 1.0 5.0)  # Sensitivity exponent grid for density scaling

# Concurrency tuning parameters to saturate cluster nodes
MAX_JOBS=8           # Number of parallel python executions
CORES_PER_JOB=4      # CPU cores (n_jobs) allocated per python process

# Create results and logging directories
mkdir -p results/logs

# Parse optional arguments to override gap types
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gap_type) GAP_TYPES=("$2"); shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Calculate total script executions
total_runs=0
for gap_type in "${GAP_TYPES[@]}"; do
    if [ "$gap_type" == "sparse" ]; then
        combos=$(( ${#RF_CONFIGS[@]} * ${#K_VALUES[@]} * ${#SPARSE_MULTIPLIERS[@]} * ${#SCALING_LAWS[@]} * ${#ALPHA_VALUES[@]} ))
        total_runs=$(( total_runs + combos ))
    else
        combos=$(( ${#RF_CONFIGS[@]} * ${#K_VALUES[@]} * ${#ALPHA_VALUES[@]} ))
        total_runs=$(( total_runs + combos ))
    fi
done

echo "=========================================================="
echo "LAUNCHING UNIFIED TOPOLOGICAL SWEEP ON CLUSTER CORES + H100"
echo "Total evaluations to run: $total_runs executions (all methods run in a single pass)"
echo "Concurrency: $MAX_JOBS parallel jobs, $CORES_PER_JOB CPU cores each"
echo "Gap Types: ${GAP_TYPES[*]}"
echo "RF Configs: ${RF_CONFIGS[*]}"
echo "K Neighbors: ${K_VALUES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "Individual run logs will be saved to: results/logs/"
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

# Pure bash semaphore to regulate process concurrency
manage_parallel_jobs() {
    while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
        sleep 0.1
    done
}

current=1
start_time=$(date +%s)

# List of approaches to run side-by-side
APPROACHES="Standard,Proximity_Baseline,Proximity_Method_A,Proximity_Method_B,Proximity_Method_C,Proximity_Method_B_C,Shaker_Likelihood_GL_Bisect"

for gap_type in "${GAP_TYPES[@]}"; do
    for config in "${RF_CONFIGS[@]}"; do
        
        # Determine loop parameters depending on gap type
        if [ "$gap_type" == "sparse" ]; then
            for law in "${SCALING_LAWS[@]}"; do
                for mult in "${SPARSE_MULTIPLIERS[@]}"; do
                    for k in "${K_VALUES[@]}"; do
                        for alpha in "${ALPHA_VALUES[@]}"; do
                            manage_parallel_jobs
                            job_id=$current
                            
                            echo "[$job_id/$total_runs] Dispatching Unified Sweep - RF=$config, K=$k, Alpha=$alpha, Gap=sparse, Law=$law, Multiplier=$mult"
                            
                            args="--approaches $APPROACHES --rf_config $config --gap_type sparse --sparse_multiplier $mult --scaling_law $law --k_neighbors $k --density_scaling_alpha $alpha --n_runs 10 --n_jobs $CORES_PER_JOB"
                            
                            ( $PYTHON_EXEC Uncertainty_Quantification.py $args > results/logs/run_${job_id}.log 2>&1 && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" || echo "   ✗ [Job $job_id/$total_runs] FAILED (check results/logs/run_${job_id}.log)" ) &
                            
                            current=$((current + 1))
                        done
                    done
                done
            done
        else
            # gap_type == "empty"
            for k in "${K_VALUES[@]}"; do
                for alpha in "${ALPHA_VALUES[@]}"; do
                    manage_parallel_jobs
                    job_id=$current
                    
                    echo "[$job_id/$total_runs] Dispatching Unified Sweep - RF=$config, K=$k, Alpha=$alpha, Gap=empty"
                    
                    args="--approaches $APPROACHES --rf_config $config --gap_type empty --k_neighbors $k --density_scaling_alpha $alpha --n_runs 10 --n_jobs $CORES_PER_JOB"
                    
                    ( $PYTHON_EXEC Uncertainty_Quantification.py $args > results/logs/run_${job_id}.log 2>&1 && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" || echo "   ✗ [Job $job_id/$total_runs] FAILED (check results/logs/run_${job_id}.log)" ) &
                    
                    current=$((current + 1))
                done
            done
        fi
        
    done
done

echo ""
echo "All jobs dispatched. Waiting for final background processes to complete..."
wait

end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo ""
echo "=========================================================="
echo "ALL UNIFIED SWEEP BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Total runtime: $((total_duration / 3600))h $(((total_duration % 3600) / 60))m $((total_duration % 60))s"
echo "=========================================================="
