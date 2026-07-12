#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=student_t_sweep
#SBATCH --output=student_t_sweep_%j.log
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Dedicated script to evaluate and compare the Normal, Student-t, and Corrected Student-t likelihood approaches

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate dyrf

# Detect the number of allocated GPUs from salloc / environment
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPUS_ARR <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS_ARR[@]}
    echo "Detected $NUM_GPUS allocated GPUs from salloc (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
else
    NUM_GPUS=4
    GPUS_ARR=("0" "1" "2" "3")
    echo "No active CUDA_VISIBLE_DEVICES found. Defaulting to 4 standard GPUs."
fi

# Exit immediately if a command exits with a non-zero status
set -e

# Define parameters for the experimental grid
RF_CONFIGS=(1 5)
SPARSE_MULTIPLIERS=(5 50)
SCALING_LAWS=("linear" "leaf")
GAP_TYPES=("empty" "sparse")

# Detect available CPU cores on node
if [ -n "$SLURM_CPUS_ON_NODE" ]; then
    TOTAL_CORES=$SLURM_CPUS_ON_NODE
else
    TOTAL_CORES=$(nproc)
fi

# Concurrency tuning parameters dynamically scaled to available GPUs and CPU cores
MAX_JOBS=$(( NUM_GPUS * 2 ))  # Concurrency: 2 processes per GPU
CORES_PER_JOB=$(( TOTAL_CORES / MAX_JOBS ))
if [ $CORES_PER_JOB -lt 1 ]; then
    CORES_PER_JOB=1
fi

SWEEP_NAME="student_t_sweep_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="results/$SWEEP_NAME"

# Create results and logging directories
mkdir -p "$SWEEP_DIR/logs"

# Calculate total script executions dynamically
total_runs=0
for gap_type in "${GAP_TYPES[@]}"; do
    if [ "$gap_type" == "sparse" ]; then
        dataset_combos=$(( ${#RF_CONFIGS[@]} * ${#SPARSE_MULTIPLIERS[@]} * ${#SCALING_LAWS[@]} ))
        total_runs=$(( total_runs + dataset_combos ))
    else
        dataset_combos=${#RF_CONFIGS[@]}
        total_runs=$(( total_runs + dataset_combos ))
    fi
done

echo "=========================================================="
echo "LAUNCHING STUDENT-T LIKELIHOOD SWEEP ON CLUSTER"
echo "Total evaluations to run: $total_runs executions"
echo "Concurrency: $MAX_JOBS parallel jobs, $CORES_PER_JOB CPU cores each"
echo "Seeds: 10"
echo "Gap Types: ${GAP_TYPES[*]}"
echo "RF Configs: ${RF_CONFIGS[*]}"
echo "Individual run logs will be saved to: $SWEEP_DIR/logs/"
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

APPROACHES="Shaker_Likelihood_Normal,Shaker_Likelihood_StudentT,Shaker_Likelihood_StudentT_Corrected"

for gap_type in "${GAP_TYPES[@]}"; do
    for config in "${RF_CONFIGS[@]}"; do
        
        # Determine loop parameters depending on gap type
        if [ "$gap_type" == "sparse" ]; then
            for law in "${SCALING_LAWS[@]}"; do
                for mult in "${SPARSE_MULTIPLIERS[@]}"; do
                     
                     manage_parallel_jobs
                     job_id=$current
                     gpu_index=$(( (current - 1) % NUM_GPUS ))
                     gpu_id=${GPUS_ARR[$gpu_index]}
                     
                     echo "[$job_id/$total_runs] Dispatching Student-t Sweep (GPU $gpu_id) - RF=$config, Gap=sparse, Law=$law, Multiplier=$mult"
                     
                     args="--approaches $APPROACHES --rf_config $config --gap_type sparse --sparse_multiplier $mult --scaling_law $law --n_runs 10 --n_jobs $CORES_PER_JOB --output_dir $SWEEP_DIR --debug_timing --ood_type manifold"
                     
                     ( CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_EXEC Uncertainty_Quantification.py $args > "$SWEEP_DIR/logs/run_${job_id}.log" 2>&1 && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" || echo "   ✗ [Job $job_id/$total_runs] FAILED (check $SWEEP_DIR/logs/run_${job_id}.log)" ) &
                     
                     current=$((current + 1))
                done
            done
        else
            # gap_type == "empty"
            manage_parallel_jobs
            job_id=$current
            gpu_index=$(( (current - 1) % NUM_GPUS ))
            gpu_id=${GPUS_ARR[$gpu_index]}
            
            echo "[$job_id/$total_runs] Dispatching Student-t Sweep (GPU $gpu_id) - RF=$config, Gap=empty"
            
            args="--approaches $APPROACHES --rf_config $config --gap_type empty --n_runs 10 --n_jobs $CORES_PER_JOB --output_dir $SWEEP_DIR --debug_timing --ood_type manifold"
            
            ( CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_EXEC Uncertainty_Quantification.py $args > "$SWEEP_DIR/logs/run_${job_id}.log" 2>&1 && echo "   ✓ [Job $job_id/$total_runs] Completed successfully" || echo "   ✗ [Job $job_id/$total_runs] FAILED (check $SWEEP_DIR/logs/run_${job_id}.log)" ) &
            
            current=$((current + 1))
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
echo "STUDENT-T LIKELIHOOD SWEEP BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Total runtime: $((total_duration / 3600))h $(((total_duration % 3600) / 60))m $((total_duration % 60))s"
echo "=========================================================="
