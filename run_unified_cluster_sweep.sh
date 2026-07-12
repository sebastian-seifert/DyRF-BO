#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=dyrf_uq_sweep
#SBATCH --output=results/dyrf_uq_sweep/logs/run_%A_%a.log
#SBATCH --error=results/dyrf_uq_sweep/logs/run_%A_%a.err
#SBATCH --array=1-250%24
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Dedicated script to evaluate baseline and proximity UQ approaches
# using a SLURM Job Array where each task processes a single function + config.

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate dyrf || true

# Setup directory structure
mkdir -p results/dyrf_uq_sweep/logs

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

# Fetch specific parameters for this array task
PARAMS_FILE="unified_sweep_params.txt"
if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameter file $PARAMS_FILE not found. Run generate_unified_sweep_params.py first."
    exit 1
fi

PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS_FILE")

if [ -z "$PARAMS" ]; then
    echo "ERROR: No parameters found at line $SLURM_ARRAY_TASK_ID in $PARAMS_FILE."
    exit 1
fi

echo "=========================================================="
echo "SLURM Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Active parameters:  $PARAMS"
echo "Using Python:       $PYTHON_EXEC"
echo "Seeds:              5"
echo "=========================================================="

# Execute single function benchmark with locked 5 seeds
$PYTHON_EXEC Uncertainty_Quantification.py \
    $PARAMS \
    --n_runs 5 \
    --output_dir results/dyrf_uq_sweep \
    --debug_timing \
    --ood_type manifold

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
