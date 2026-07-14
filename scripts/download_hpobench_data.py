#!/usr/bin/env python3
import sys

try:
    from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
except ImportError:
    print("ERROR: HPOBench is not installed. Please run 'pip install --ignore-requires-python -r requirements.txt' first.")
    sys.exit(1)

models = ['svm', 'xgb', 'rf']

print("==================================================")
print("STARTING HPOBENCH OFFLINE DATASETS DOWNLOAD & CACHING")
print("==================================================")

for model in models:
    try:
        print(f"Caching {model} dataset bundle...")
        # Instantiating tabular benchmark with model and task_id=3 triggers the full model zip download
        TabularBenchmark(model=model, task_id=3)
        print(f"✓ Successfully cached {model} bundle!")
    except Exception as e:
        print(f"✗ Error caching {model}: {e}")

print("==================================================")
print("DATASETS CACHING COMPLETED")
print("==================================================")
