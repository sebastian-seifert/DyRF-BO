#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=student_t_sweep
#SBATCH --output=results/student_t_sweep/logs/run_%A_%a.log
#SBATCH --error=results/student_t_sweep/logs/run_%A_%a.err
#SBATCH --array=1-410
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Dedicated script to evaluate the Normal, Student-t, and Corrected Student-t likelihood approaches
# using a SLURM Job Array where each task processes a single function + config.

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate dyrf || true

# Setup directory structure
mkdir -p results/student_t_sweep/logs

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
PARAMS_FILE="sweep_params.txt"
if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: Parameter file $PARAMS_FILE not found. Run generate_sweep_params.py first."
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
echo "=========================================================="

APPROACHES="Shaker_Likelihood_Normal,Shaker_Likelihood_StudentT,Shaker_Likelihood_StudentT_Corrected"

# Execute single function benchmark
$PYTHON_EXEC Uncertainty_Quantification.py \
    $PARAMS \
    --approaches $APPROACHES \
    --n_runs 10 \
    --output_dir results/student_t_sweep \
    --debug_timing \
    --ood_type manifold

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
