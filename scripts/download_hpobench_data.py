#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Override data directories for CARPS/YAHPO and HPOBench
os.environ["CARPS_TASK_DATA_DIR"] = "/bigwork/nhwpseis/benchmarks"

try:
    import hpobench
    hpobench.config_file.data_dir = Path("/bigwork/nhwpseis/benchmarks/hpobench")
    from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
    from hpobench.util.data_manager import SurrogateSVMDataManager
except ImportError:



    print("ERROR: HPOBench is not installed. Please run 'pip install --ignore-requires-python -r requirements.txt' first.")
    sys.exit(1)

print("==================================================")
print("STARTING HPOBENCH OFFLINE DATASETS DOWNLOAD & CACHING")
print("==================================================")

# 1. Tabular ML datasets
models = ['svm', 'xgb', 'rf', 'lr']
for model in models:
    try:
        print(f"Caching Tabular ML {model} dataset bundle...")
        # Instantiating tabular benchmark with model and task_id=3 triggers the full model zip download
        TabularBenchmark(model=model, task_id=3)
        print(f"✓ Successfully cached {model} bundle!")
    except Exception as e:
        print(f"✗ Error caching {model}: {e}")

# 2. Surrogate datasets
try:
    print("\nCaching Surrogate dataset bundle...")
    # Loading SurrogateSVMDataManager triggers download of the single surrogates.tar.gz
    # which contains all surrogates (SVM and ParamNet: adult, higgs, letter, mnist, optdigits, poker, vehicle)
    dm = SurrogateSVMDataManager()
    dm.load()
    print("✓ Successfully cached Surrogate bundle!")
except Exception as e:
    print(f"✗ Error caching Surrogate bundle: {e}")

# 3. NAS-Bench-201 Singularity containers
try:
    print("\nCaching NAS-Bench-201 Singularity container...")
    from hpobench.container.benchmarks.nas.nasbench_201 import ImageNetNasBench201Benchmark
    ImageNetNasBench201Benchmark()
    print("✓ Successfully cached NAS-Bench-201 Singularity container!")
except Exception as e:
    print(f"✗ Error caching NAS-Bench-201 container: {e}")

print("==================================================")
print("DATASETS CACHING COMPLETED")
print("==================================================")
