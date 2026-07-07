#!/bin/bash

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate dyrf

# Detect the number of allocated GPUs from salloc / environment
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Split comma-separated list to count the visible devices (can be integers or MIG UUIDs)
    IFS=',' read -ra GPUS_ARR <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS_ARR[@]}
    echo "Detected $NUM_GPUS allocated GPUs/MIG-slices from salloc (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
else
    NUM_GPUS=4
    GPUS_ARR=("0" "1" "2" "3")
    echo "No active CUDA_VISIBLE_DEVICES found. Defaulting to 4 standard GPUs."
fi


# Exit immediately if a command exits with a non-zero status
set -e

# Define parameters for the experimental grid
RF_CONFIGS=(1 3 5)
K_VALUES=(20 100 500)
SPARSE_MULTIPLIERS=(5 15 50)
SCALING_LAWS=("linear" "leaf")
GAP_TYPES=("empty" "sparse")
ALPHA_VALUES=(0.1 1.0 5.0)  # Sensitivity exponent grid for density scaling

# Detect available CPU cores on node
if [ -n "$SLURM_CPUS_ON_NODE" ]; then
    TOTAL_CORES=$SLURM_CPUS_ON_NODE
else
    TOTAL_CORES=$(nproc)
fi

# Concurrency tuning parameters dynamically scaled to available GPUs and CPU cores
MAX_JOBS=$(( NUM_GPUS * 2 ))  # Saturation factor of 2 processes per GPU
CORES_PER_JOB=$(( TOTAL_CORES / MAX_JOBS ))
if [ $CORES_PER_JOB -lt 1 ]; then
    CORES_PER_JOB=1
fi

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

# Calculate total script executions (1 baseline + 9 grid combinations per dataset)
total_runs=0
for gap_type in "${GAP_TYPES[@]}"; do
    if [ "$gap_type" == "sparse" ]; then
        dataset_combos=$(( ${#RF_CONFIGS[@]} * ${#SPARSE_MULTIPLIERS[@]} * ${#SCALING_LAWS[@]} ))
        total_runs=$(( total_runs + (dataset_combos * 10) ))
    else
        dataset_combos=${#RF_CONFIGS[@]}
        total_runs=$(( total_runs + (dataset_combos * 10) ))
    fi
done

echo "=========================================================="
echo "LAUNCHING UNIFIED TOPOLOGICAL SWEEP ON CLUSTER CORES + H100"
echo "Total evaluations to run: $total_runs executions (split for zero-overhead baselines)"
echo "Concurrency: $MAX_JOBS parallel jobs, $CORES_PER_JOB CPU cores each"
echo "Gap Types: ${GAP_TYPES[*]}"
echo "RF Configs: ${RF_CONFIGS[*]}"
echo "K Neighbors: ${K_VALUES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo "Individual run logs will be saved to: results/logs/"
echo "=========================================================="

# Detect active Python environment (prefer active conda environment first)
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    PYTHON_EXEC="python"
elif [ -f ".venv/bin/python" ]; then
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

# UQ Approach groupings to prevent redundant baseline calculations
BASELINES="Standard,Chen,Shaker_GMM_Entropy,Shaker_Likelihood_GL_Bisect,Shaker_Likelihood_GL_Newton,Shaker_Likelihood_Trapz_Bisect,Shaker_Likelihood_Trapz_Newton"
PROXIMITY_METHODS="Proximity_Baseline,Proximity_Method_A,Proximity_Method_B,Proximity_Method_C,Proximity_Method_B_C"

for gap_type in "${GAP_TYPES[@]}"; do
    for config in "${RF_CONFIGS[@]}"; do
        
        # Determine loop parameters depending on gap type
        if [ "$gap_type" == "sparse" ]; then
            for law in "${SCALING_LAWS[@]}"; do
                for mult in "${SPARSE_MULTIPLIERS[@]}"; do
                    
                    # 1. Dispatch Baseline Job (Once per dataset combo)
                    manage_parallel_jobs
                    job_id=$current
                    gpu_index=$(( (current - 1) % NUM_GPUS ))
                    gpu_id=${GPUS_ARR[$gpu_index]}
                    
                    echo "[$job_id/$total_runs] Dispatching Baseline Sweep (GPU $gpu_id) - RF=$config, Gap=sparse, Law=$law, Multiplier=$mult"
                    
                    args="--approaches $BASELINES --rf_config $config --gap_type sparse --sparse_multiplier $mult --scaling_law $law --n_runs 10 --n_jobs $CORES_PER_JOB"
                    
                    ( CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_EXEC Uncertainty_Quantification.py $args > results/logs/run_${job_id}.log 2>&1 && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" || echo "   ✗ [Job $job_id/$total_runs] FAILED (check results/logs/run_${job_id}.log)" ) &
                    
                    current=$((current + 1))
                    
                    # 2. Dispatch Proximity Sweep (Loops over K and Alpha)
                    for k in "${K_VALUES[@]}"; do
                        for alpha in "${ALPHA_VALUES[@]}"; do
                            manage_parallel_jobs
                            job_id=$current
                            gpu_index=$(( (current - 1) % NUM_GPUS ))
                            gpu_id=${GPUS_ARR[$gpu_index]}
                            
                            echo "[$job_id/$total_runs] Dispatching Proximity Sweep (GPU $gpu_id) - RF=$config, K=$k, Alpha=$alpha, Gap=sparse, Law=$law, Multiplier=$mult"
                            
                            args="--approaches $PROXIMITY_METHODS --rf_config $config --gap_type sparse --sparse_multiplier $mult --scaling_law $law --k_neighbors $k --density_scaling_alpha $alpha --n_runs 10 --n_jobs $CORES_PER_JOB"
                            
                            ( CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_EXEC Uncertainty_Quantification.py $args > results/logs/run_${job_id}.log 2>&1 && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" || echo "   ✗ [Job $job_id/$total_runs] FAILED (check results/logs/run_${job_id}.log)" ) &
                            
                            current=$((current + 1))
                        done
                    done
                done
            done
        else
            # gap_type == "empty"
            
            # 1. Dispatch Baseline Job (Once per dataset combo)
            manage_parallel_jobs
            job_id=$current
            gpu_index=$(( (current - 1) % NUM_GPUS ))
            gpu_id=${GPUS_ARR[$gpu_index]}
            
            echo "[$job_id/$total_runs] Dispatching Baseline Sweep (GPU $gpu_id) - RF=$config, Gap=empty"
            
            args="--approaches $BASELINES --rf_config $config --gap_type empty --n_runs 10 --n_jobs $CORES_PER_JOB"
            
            ( CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_EXEC Uncertainty_Quantification.py $args > results/logs/run_${job_id}.log 2>&1 && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" || echo "   ✗ [Job $job_id/$total_runs] FAILED (check results/logs/run_${job_id}.log)" ) &
            
            current=$((current + 1))
            
            # 2. Dispatch Proximity Sweep (Loops over K and Alpha)
            for k in "${K_VALUES[@]}"; do
                for alpha in "${ALPHA_VALUES[@]}"; do
                    manage_parallel_jobs
                    job_id=$current
                    gpu_index=$(( (current - 1) % NUM_GPUS ))
                    gpu_id=${GPUS_ARR[$gpu_index]}
                    
                    echo "[$job_id/$total_runs] Dispatching Proximity Sweep (GPU $gpu_id) - RF=$config, K=$k, Alpha=$alpha, Gap=empty"
                    
                    args="--approaches $PROXIMITY_METHODS --rf_config $config --gap_type empty --k_neighbors $k --density_scaling_alpha $alpha --n_runs 10 --n_jobs $CORES_PER_JOB"
                    
                    ( CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_EXEC Uncertainty_Quantification.py $args > results/logs/run_${job_id}.log 2>&1 && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" || echo "   ✗ [Job $job_id/$total_runs] FAILED (check results/logs/run_${job_id}.log)" ) &
                    
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
