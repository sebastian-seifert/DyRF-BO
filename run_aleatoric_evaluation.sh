#!/bin/bash
#SBATCH --job-name=aleatoric_eval
#SBATCH --output=aleatoric_%j.log
#SBATCH --gres=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Dedicated script to evaluate the quality of aleatoric uncertainty estimations

# Initialize Conda and activate environment
eval "$(conda shell.bash hook)"
conda activate dyrf

# Detect active Python environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    PYTHON_EXEC="python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
else
    PYTHON_EXEC="python"
fi

echo "=================================================="
echo "RUNNING ALEATORIC UNCERTAINTY QUALITY EVALUATION"
echo "Using python executable: $PYTHON_EXEC"
echo "=================================================="

$PYTHON_EXEC evaluate_aleatoric.py "$@"

echo "=================================================="
echo "EVALUATION COMPLETED SUCCESSFULLY!"
echo "=================================================="
