#!/bin/bash

# Configuration settings
SEEDS=(1 2 3 4 5)
TRIALS=50
TASKS=(
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_3"
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_12"
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_31"
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_3"
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_12"
    "+task/HPOBench/blackbox/tabular/ml=cfg_ml_xgboost_31"
)

APPROACHES=(
    "standard_disagreement"
    "chen_variance"
    "shaker_entropy"
    "likelihood_credal"
    "standard_proximity"
    "proximity_b"
    "proximity_bc"
)

echo "=========================================================="
echo "DISPATCHING FULL HPOBENCH COMPARISON SWEEP"
echo "=========================================================="

for task in "${TASKS[@]}"; do
    # Extract config name from string for clean log naming
    task_name=$(echo "$task" | grep -oE "cfg_ml_[a-z0-9_]+")
    
    for approach in "${APPROACHES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Submitting: $approach on $task_name (Seed $seed)"
            
            # Submit each run to the cluster using sbatch
            sbatch scripts/submit_hpobench_carps_sweep.sbatch \
                +optimizer=dyrf_epistemic_hpobench \
                optimizer.extractor_name="$approach" \
                "$task" \
                task.optimization_resources.n_trials=$TRIALS \
                seed=$seed \
                optimizer.telemetry_path="results/telemetry_${approach}_${task_name}_seed${seed}.json"
        done
    done
done

echo "=========================================================="
echo "All comparison runs submitted to Slurm!"
echo "=========================================================="
