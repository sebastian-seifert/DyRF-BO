#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Define parameters for the experimental grid (moderately downscaled for efficiency)
RF_CONFIGS=(1 3 5)
K_VALUES=(20 100 500)
SPARSE_MULTIPLIERS=(5 15 50)
SCALING_LAWS=("linear" "leaf")
GAP_TYPES=("empty" "sparse")
ALPHA_VALUES=(0.1 1.0 5.0)  # Sensitivity exponent grid for Method C and B+C

# Concurrency tuning parameters to saturate 32 CPU cores & H100 GPU
MAX_JOBS=8           # Number of parallel python executions
CORES_PER_JOB=4      # CPU cores (n_jobs) allocated per python process (8x4 = 32 cores total)

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

# Calculate total script executions (highly optimized to avoid redundant 'k_neighbors=auto' loops)
total_runs=0
for gap_type in "${GAP_TYPES[@]}"; do
    if [ "$gap_type" == "sparse" ]; then
        # Number of unique RF config x sparse config combinations
        combos=$(( ${#RF_CONFIGS[@]} * ${#SPARSE_MULTIPLIERS[@]} * ${#SCALING_LAWS[@]} ))
        
        # 1. Baseline: 3 (K) x combos
        # 2. Method A (TNS): 3 (K) x combos
        # 3. Method B (TWQ): 1 (auto) x combos
        # 4. Method C (TDS): 3 (K) x 3 (alpha) x combos
        # 5. Method B+C (TWQ+TDS): 1 (auto) x 3 (alpha) x combos
        total_runs=$(( total_runs + combos * (3 + 3 + 1 + 3 * 3 + 1 * 3) ))
    else
        # gap_type == "empty"
        combos=${#RF_CONFIGS[@]}
        total_runs=$(( total_runs + combos * (3 + 3 + 1 + 3 * 3 + 1 * 3) ))
    fi
done

echo "=========================================================="
echo "LAUNCHING PARALLEL TOPOLOGICAL SWEEP ON 64 CORES + H100 GPU"
echo "Total evaluations to run: $total_runs executions"
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

for gap_type in "${GAP_TYPES[@]}"; do
    for config in "${RF_CONFIGS[@]}"; do
        
        # Determine loop parameters depending on gap type
        if [ "$gap_type" == "sparse" ]; then
            for law in "${SCALING_LAWS[@]}"; do
                for mult in "${SPARSE_MULTIPLIERS[@]}"; do
                    for k in "${K_VALUES[@]}"; do
                        
                        common_args="--rf_config $config --gap_type sparse --sparse_multiplier $mult --scaling_law $law --n_runs 5 --n_jobs $CORES_PER_JOB"
                        
                        # 1. Baseline: Standard Proximity (No density scaling, no alpha loop)
                        manage_parallel_jobs
                        echo "[$current/$total_runs] Launching Baseline (Standard) - RF=$config, K=$k, Gap=sparse, Law=$law, Multiplier=$mult"
                        $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "$k" > results/logs/run_${current}.log 2>&1 &
                        current=$((current + 1))
                        
                        # 2. Method A: Topological Neighbor Selection (TNS) (No density scaling, no alpha loop)
                        manage_parallel_jobs
                        echo "[$current/$total_runs] Launching Method A (TNS) - RF=$config, K=$k, Lambda=1.0, Gap=sparse, Law=$law, Multiplier=$mult"
                        $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "$k" --topological_decay_lambda 1.0 > results/logs/run_${current}.log 2>&1 &
                        current=$((current + 1))
                        
                        # 3. Method B: Topological Weighted Quantiles (TWQ) - Only run once per K loop (uses auto neighbors, no alpha loop)
                        if [ "$k" == "20" ]; then
                            manage_parallel_jobs
                            echo "[$current/$total_runs] Launching Method B (TWQ) - RF=$config, K=auto, Lambda=1.0, Gap=sparse, Law=$law, Multiplier=$mult"
                            $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "auto" --topological_decay_lambda 1.0 > results/logs/run_${current}.log 2>&1 &
                            current=$((current + 1))
                        fi
                        
                        # 4. Method C: Topological Density Scaling (TDS) (Requires alpha loop)
                        for alpha in "${ALPHA_VALUES[@]}"; do
                            manage_parallel_jobs
                            echo "[$current/$total_runs] Launching Method C (TDS) - RF=$config, K=$k, Lambda=5.0, Alpha=$alpha, Gap=sparse, Law=$law, Multiplier=$mult"
                            $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "$k" --topological_decay_lambda 5.0 --use_density_scaling --density_scaling_alpha "$alpha" > results/logs/run_${current}.log 2>&1 &
                            current=$((current + 1))
                        done
                        
                        # 5. Method B+C: Topological Weighted Quantiles + Density Scaling - Only run once per K loop (Requires alpha loop)
                        if [ "$k" == "20" ]; then
                            for alpha in "${ALPHA_VALUES[@]}"; do
                                manage_parallel_jobs
                                echo "[$current/$total_runs] Launching Method B+C (TWQ+TDS) - RF=$config, K=auto, Lambda=5.0, Alpha=$alpha, Gap=sparse, Law=$law, Multiplier=$mult"
                                $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "auto" --topological_decay_lambda 5.0 --use_density_scaling --density_scaling_alpha "$alpha" > results/logs/run_${current}.log 2>&1 &
                                current=$((current + 1))
                            done
                        fi
                        
                    done
                done
            done
        else
            # gap_type == "empty"
            for k in "${K_VALUES[@]}"; do
                
                common_args="--rf_config $config --gap_type empty --n_runs 5 --n_jobs $CORES_PER_JOB"
                
                # 1. Baseline: Standard Proximity
                manage_parallel_jobs
                echo "[$current/$total_runs] Launching Baseline (Standard) - RF=$config, K=$k, Gap=empty"
                $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "$k" > results/logs/run_${current}.log 2>&1 &
                current=$((current + 1))
                
                # 2. Method A: Topological Neighbor Selection (TNS)
                manage_parallel_jobs
                echo "[$current/$total_runs] Launching Method A (TNS) - RF=$config, K=$k, Lambda=1.0, Gap=empty"
                $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "$k" --topological_decay_lambda 1.0 > results/logs/run_${current}.log 2>&1 &
                current=$((current + 1))
                
                # 3. Method B: Topological Weighted Quantiles (TWQ) - Only run once per K loop (uses auto neighbors)
                if [ "$k" == "20" ]; then
                    manage_parallel_jobs
                    echo "[$current/$total_runs] Launching Method B (TWQ) - RF=$config, K=auto, Lambda=1.0, Gap=empty"
                    $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "auto" --topological_decay_lambda 1.0 > results/logs/run_${current}.log 2>&1 &
                    current=$((current + 1))
                fi
                
                # 4. Method C: Topological Density Scaling (TDS) (Requires alpha loop)
                for alpha in "${ALPHA_VALUES[@]}"; do
                    manage_parallel_jobs
                    echo "[$current/$total_runs] Launching Method C (TDS) - RF=$config, K=$k, Lambda=5.0, Alpha=$alpha, Gap=empty"
                    $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "$k" --topological_decay_lambda 5.0 --use_density_scaling --density_scaling_alpha "$alpha" > results/logs/run_${current}.log 2>&1 &
                    current=$((current + 1))
                done
                
                # 5. Method B+C: Topological Weighted Quantiles + Density Scaling - Only run once per K loop (Requires alpha loop)
                if [ "$k" == "20" ]; then
                    for alpha in "${ALPHA_VALUES[@]}"; do
                        manage_parallel_jobs
                        echo "[$current/$total_runs] Launching Method B+C (TWQ+TDS) - RF=$config, K=auto, Lambda=5.0, Alpha=$alpha, Gap=empty"
                        $PYTHON_EXEC Uncertainty_Quantification.py $common_args --k_neighbors "auto" --topological_decay_lambda 5.0 --use_density_scaling --density_scaling_alpha "$alpha" > results/logs/run_${current}.log 2>&1 &
                        current=$((current + 1))
                    done
                fi
                
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
echo "ALL TOPOLOGICAL SWEEP BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Total runtime: $((total_duration / 3600))h $(((total_duration % 3600) / 60))m $((total_duration % 60))s"
echo "=========================================================="
