#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=ood_dynamic_lambda
#SBATCH --output=results/ood_dynamic_lambda_%j.log
#SBATCH --error=results/ood_dynamic_lambda_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Slurm job for 1-15D Pure OOD Performance Benchmark (GPU accelerated).

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dyrf 2>/dev/null || true

export PYTHONPATH=.

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID | Running Pure OOD Performance Evaluation on GPU"
echo "=================================================="

python3 scripts/evaluate_ood_dynamic_lambda.py --output_dir results/ood_dynamic_lambda --device auto

echo "=================================================="
echo "Pure OOD Benchmark Finished Successfully!"
echo "=================================================="
