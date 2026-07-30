#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so carps_integration can be imported reliably
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Override data directories for CARPS/YAHPO and HPOBench if running on cluster
if os.path.exists("/bigwork/nhwpseis/benchmarks"):
    os.environ["CARPS_TASK_DATA_DIR"] = "/bigwork/nhwpseis/benchmarks"

    import hpobench
    hpobench.config_file.data_dir = Path("/bigwork/nhwpseis/benchmarks/hpobench")

    import carps.objective_functions.yahpo
    carps.objective_functions.yahpo.YAHPO_TASK_DATA_DIR = Path("/bigwork/nhwpseis/benchmarks/yahpo-data")

    import yahpo_gym
    try:
        yahpo_gym.local_config.init_config(data_path="/bigwork/nhwpseis/benchmarks/yahpo-data")
    except Exception:
        pass

import ConfigSpace
if not hasattr(ConfigSpace.ConfigurationSpace, "_sort_hyperparameters"):
    def _sort_hyperparameters_compat(self):
        return list(self.values()) if hasattr(self, "values") else self.get_hyperparameters()
    ConfigSpace.ConfigurationSpace._sort_hyperparameters = _sort_hyperparameters_compat

import carps.utils.loggingutils

# Patch log_python_env to ensure parent directories exist
original_log_python_env = carps.utils.loggingutils.log_python_env

def patched_log_python_env(log_file="env_log.txt"):
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    original_log_python_env(log_file=log_file_path)

carps.utils.loggingutils.log_python_env = patched_log_python_env

import carps.loggers.file_logger
import carps.loggers.abstract_logger
from carps.loggers.file_logger import get_run_directory, logger

def safe_file_logger_init(self, overwrite: bool = False, directory=None):
    carps.loggers.abstract_logger.AbstractLogger.__init__(self)

    directory = Path(directory) if directory is not None else get_run_directory()
    assert directory is not None, "Directory must be specified in FileLogger or hydra run dir must be available."
    self.directory = directory
    
    filename = getattr(self, "_filename", "trial_logs.jsonl")
    if (directory / filename).is_file():
        if overwrite:
            logger.info(f"Found previous run. Removing '{directory}'.")
            for root, _dirs, files in os.walk(directory):
                for f in files:
                    full_fn = Path(root) / f
                    if ".hydra" not in str(full_fn):
                        try:
                            Path(full_fn).unlink()
                            logger.debug(f"Removed {full_fn}")
                        except (FileNotFoundError, OSError):
                            pass
        else:
            raise RuntimeError(
                f"Found previous run at '{directory}'. Stopping run. If you want to overwrite, specify overwrite "
                f"for the file logger in the config (CARP-S/carps/configs/logger.yaml)."
            )

    carps.utils.loggingutils.log_python_env(log_file=Path(directory) / "env_info.txt")

carps.loggers.file_logger.FileLogger.__init__ = safe_file_logger_init




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

import sys
import omegaconf

# Monkey-patch OmegaConf.select to prevent Hydra's exception formatter from crashing
# on MissingMandatoryValue errors when resolving output directories.
original_select = omegaconf.OmegaConf.select

def patched_select(cfg, key, *args, **kwargs):
    try:
        return original_select(cfg, key, *args, **kwargs)
    except Exception as e:
        # If it's a key in Hydra and failed to resolve due to missing mandatory values,
        # return a fallback directory/string path to allow execution or logging to proceed.
        if key in ("hydra.run.dir", "hydra.sweep.dir", "hydra.sweep.subdir"):
            return "runs/unknown_optimizer/unknown_benchmark/unknown_task/1"
        if key == "conda_env_name":
            return "carps_env"
        if key in ("benchmark_id", "task_id", "optimizer_id", "optimizer_container_id"):
            return f"unknown_{key}"
        raise e

omegaconf.OmegaConf.select = patched_select

# Patch OmegaConf.to_container to prevent InterpolationToMissingValueError during container conversion
original_to_container = omegaconf.OmegaConf.to_container

def safe_to_container(cfg, resolve=True, **kwargs):
    try:
        return original_to_container(cfg, resolve=resolve, **kwargs)
    except Exception:
        for key in ["benchmark_id", "task_id", "optimizer_id", "optimizer_container_id"]:
            try:
                val = omegaconf.OmegaConf.select(cfg, key)
                if val is None or val == "???":
                    setattr(cfg, key, f"unknown_{key}")
            except Exception:
                setattr(cfg, key, f"unknown_{key}")
        try:
            val = omegaconf.OmegaConf.select(cfg, "conda_env_name")
            if val is None or val == "???":
                cfg.conda_env_name = "carps_env"
        except Exception:
            cfg.conda_env_name = "carps_env"
        try:
            return original_to_container(cfg, resolve=resolve, **kwargs)
        except Exception:
            return original_to_container(cfg, resolve=False, **kwargs)

omegaconf.OmegaConf.to_container = safe_to_container

# Monkey-patch SMAC3Optimizer to accept extra kwargs like acq_func_name and configure acquisition function
import carps.optimizers.smac20
original_smac3_init = carps.optimizers.smac20.SMAC3Optimizer.__init__

def patched_smac3_init(self, task, smac_cfg, loggers=None, expects_multiple_objectives=False, expects_fidelities=False, **kwargs):
    if "acq_func_name" in kwargs:
        self.acq_func_name = kwargs.pop("acq_func_name")
    original_smac3_init(
        self,
        task=task,
        smac_cfg=smac_cfg,
        loggers=loggers,
        expects_multiple_objectives=expects_multiple_objectives,
        expects_fidelities=expects_fidelities,
    )

carps.optimizers.smac20.SMAC3Optimizer.__init__ = patched_smac3_init

def patched_smac3_setup_optimizer(self):
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from smac.scenario import Scenario
    from carps.optimizers.smac20 import maybe_inst_add_scenario

    smac_class = get_class(self.smac_cfg.smac_class)
    if smac_class == get_class("smac.facade.multi_fidelity_facade.MultiFidelityFacade"):
        self.fidelity_enabled = True

    smac_kwargs = {}
    if self.smac_cfg.smac_kwargs is not None:
        smac_kwargs = OmegaConf.to_container(self.smac_cfg.smac_kwargs, resolve=True, enum_to_str=True)

    if hasattr(self, "acq_func_name") and self.acq_func_name:
        import smac.acquisition.function as acq_module
        acq_map = {
            "ei": acq_module.EI,
            "pi": acq_module.PI,
            "lcb": acq_module.LCB,
        }
        acq_cls = acq_map.get(str(self.acq_func_name).lower())
        if acq_cls is not None:
            smac_kwargs["acquisition_function"] = acq_cls()

    scenario_kwargs = {
        "configspace": self.configspace,
    }
    _scenario_kwargs = OmegaConf.to_container(self.smac_cfg.scenario, resolve=True)
    scenario_kwargs.update(_scenario_kwargs)

    scenario = Scenario(**scenario_kwargs)
    smac_kwargs["scenario"] = scenario

    if "callbacks" not in smac_kwargs:
        smac_kwargs["callbacks"] = []
    elif "callbacks" in smac_kwargs and isinstance(smac_kwargs["callbacks"], dict):
        smac_kwargs["callbacks"] = list(smac_kwargs["callbacks"].values())

    smac_kwargs = maybe_inst_add_scenario(smac_kwargs, "intensifier", scenario)

    if (
        "acquisition_function" in smac_kwargs
        and "acquisition_maximizer" in smac_kwargs
    ):
        smac_kwargs["acquisition_maximizer"] = smac_kwargs["acquisition_maximizer"](
            configspace=self.configspace, acquisition_function=smac_kwargs["acquisition_function"]
        )
        if hasattr(smac_kwargs["acquisition_maximizer"], "selector") and hasattr(
            smac_kwargs["acquisition_maximizer"].selector, "expl2callback"
        ):
            smac_kwargs["callbacks"].append(smac_kwargs["acquisition_maximizer"].selector.expl2callback)

    smac_kwargs = maybe_inst_add_scenario(smac_kwargs, "config_selector", scenario)
    smac_kwargs = maybe_inst_add_scenario(smac_kwargs, "initial_design", scenario)
    smac_kwargs = maybe_inst_add_scenario(smac_kwargs, "multi_objective_algorithm", scenario)

    smac = smac_class(
        target_function=self.target_function,
        **smac_kwargs,
    )
    return smac

carps.optimizers.smac20.SMAC3Optimizer._setup_optimizer = patched_smac3_setup_optimizer

from carps.run import main

if __name__ == "__main__":
    main()
