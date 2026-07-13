#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=hybrid_sweep
#SBATCH --output=results/hybrid_sweep/sweep_%j.log
#SBATCH --error=results/hybrid_sweep/sweep_%j.err
#SBATCH --gres=gpu:a100:6
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# Dedicated script to evaluate the Hybrid Proximity-Epistemic approaches
# Optimized for parallel execution across 4 A100 GPUs in a single job allocation.

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate dyrf || true

# Setup directory structure
mkdir -p results/hybrid_sweep/logs

# Resolve Python executable
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_EXEC="../.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
else
    PYTHON_EXEC="python"
fi

# Fetch specific parameters for the sweep tasks
PARAMS_FILE="hybrid_sweep_params.txt"
if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameter file $PARAMS_FILE not found. Run generate_hybrid_sweep_params.py first."
    exit 1
fi

# Detect GPU layout
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPUS_ARR <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS_ARR[@]}
    echo "Detected $NUM_GPUS allocated GPUs from salloc (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
else
    GPUS_ARR=(0 1 2 3)
    NUM_GPUS=4
    echo "No active CUDA_VISIBLE_DEVICES found. Defaulting to 4 GPUs layout."
fi

# Max concurrent jobs: 1 per GPU
MAX_JOBS=$NUM_GPUS
echo "Running up to $MAX_JOBS parallel jobs concurrently..."

APPROACHES="Hybrid_Shaker_Entropy_L20,Hybrid_Shaker_Entropy_L40,Hybrid_Shaker_Entropy_L70,Hybrid_Likelihood_L20,Hybrid_Likelihood_L40,Hybrid_Likelihood_L70"
total_runs=$(wc -l < "$PARAMS_FILE")
current=1

while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi

    # Limit the number of concurrent processes
    while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done

    # Round-robin GPU assignment
    gpu_index=$(( (current - 1) % NUM_GPUS ))
    gpu_id=${GPUS_ARR[$gpu_index]}

    echo "[$current/$total_runs] Dispatching to GPU $gpu_id: $line"

    # Run in background and redirect output to specific log file
    ( CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_EXEC Uncertainty_Quantification.py \
        $line \
        --approaches $APPROACHES \
        --n_runs 5 \
        --output_dir results/hybrid_sweep \
        --debug_timing \
        --ood_type manifold > results/hybrid_sweep/logs/run_${current}.log 2>&1 \
        && echo "   ✓ [Job $current/$total_runs] Completed successfully" \
        || echo "   ✗ [Job $current/$total_runs] FAILED (check results/hybrid_sweep/logs/run_${current}.log)" ) &

    current=$((current + 1))
done < "$PARAMS_FILE"

# Wait for all background jobs to finish
wait

echo "=========================================================="
echo "All $total_runs hybrid UQ sweep jobs completed."
echo "=========================================================="
