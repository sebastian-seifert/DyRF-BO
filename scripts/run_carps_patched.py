import os
from pathlib import Path

# Override data directories for CARPS/YAHPO and HPOBench
os.environ["CARPS_TASK_DATA_DIR"] = "/bigwork/nhwpseis/benchmarks"

import hpobench
hpobench.config_file.data_dir = Path("/bigwork/nhwpseis/benchmarks/hpobench")

import carps.objective_functions.yahpo
carps.objective_functions.yahpo.YAHPO_TASK_DATA_DIR = Path("/bigwork/nhwpseis/benchmarks/yahpo-data")

import carps.utils.loggingutils
# Patch log_python_env to ensure parent directories exist
original_log_python_env = carps.utils.loggingutils.log_python_env

def patched_log_python_env(log_file="env_log.txt"):
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    original_log_python_env(log_file=log_file_path)

carps.utils.loggingutils.log_python_env = patched_log_python_env



import argparse


# Monkey-patch argparse.ArgumentParser._check_help to bypass compatibility crashes
# with older Hydra-Core (LazyCompletionHelp) under Python 3.14.
if hasattr(argparse.ArgumentParser, "_check_help"):
    original_check_help = argparse.ArgumentParser._check_help
    def custom_check_help(self, action):
        try:
            original_check_help(self, action)
        except Exception:
            pass
    argparse.ArgumentParser._check_help = custom_check_help

from carps.run import main

if __name__ == "__main__":
    main()
