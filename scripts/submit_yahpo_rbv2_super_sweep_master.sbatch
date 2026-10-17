#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=supr_master
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=results/sweep_yahpo_rbv2_super/logs/master_orchestrator_%j.out
#SBATCH --error=results/sweep_yahpo_rbv2_super/logs/master_orchestrator_%j.err

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p results/sweep_yahpo_rbv2_super/logs

# Resolve Python environment
if [ -f "/bigwork/nhwpseis/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="/bigwork/nhwpseis/.conda/envs/dyrf/bin/python"
elif [ -f "$HOME/.conda/envs/dyrf/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/dyrf/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="$(which python3 || which python)"
fi

exec "$PYTHON_BIN" scripts/master_slurm_scheduler.py \
    --manifest results/sweep_yahpo_rbv2_super/tasks.txt \
    --sbatch scripts/submit_yahpo_rbv2_super_sweep_array.sbatch \
    --chunk-size 250 \
    --max-tasks 4500 \
    --trigger-jobs 10 \
    --concurrency 25 \
    --state-file results/sweep_yahpo_rbv2_super/scheduler_state.json \
    --job-prefix supr \
    "$@"
