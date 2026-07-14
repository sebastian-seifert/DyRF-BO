from __future__ import annotations
import os
import json
import numpy as np
from typing import Any, List, Tuple
from ConfigSpace import Configuration, ConfigurationSpace

from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.task import Task
from carps.utils.types import Incumbent, SearchSpace
from rf_dynamic.dynamic_rf_surrogate import DynamicRFSurrogate

class CARPSDynamicRFOptimizer(Optimizer):
    """
    CARP-S compatible Bayesian Optimization optimizer wrapper that uses
    Dynamic RF surrogate models with Extensible Epistemic UQ Extractors.
    """
    def __init__(
        self,
        task: Task,
        loggers: list[Any] | None = None,
        extractor_name: str = "standard_disagreement",
        n_init: int = 10,
        kappa: float = 1.96,
        telemetry_path: str = "dyrf_bo_telemetry.json",
        window_size: int = 5,
        n_base: int = 100,
        n_min: int = 10,
        n_max: int = 200,
        gamma: float = 1.0,
        depth_base: int = 12,
        depth_min: int = 5,
        depth_max: int = 30,
        beta: float = 5.0,
        extractor_kwargs: dict | None = None,
        rf_kwargs: dict | None = None
    ) -> None:
        super().__init__(
            task=task,
            loggers=loggers,
            expects_multiple_objectives=False,
            expects_fidelities=False
        )
        
        self.configspace: ConfigurationSpace = self.task.objective_function.configspace
        self.extractor_name = extractor_name
        self.n_init = n_init
        self.kappa = kappa
        self.telemetry_path = telemetry_path
        
        # Dynamic RF hyperparameters
        self.window_size = window_size
        self.n_base = n_base
        self.n_min = n_min
        self.n_max = n_max
        self.gamma = gamma
        self.depth_base = depth_base
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.beta = beta
        self.extractor_kwargs = extractor_kwargs or {}
        self.rf_kwargs = rf_kwargs or {}
        
        self.history: List[Tuple[TrialInfo, TrialValue]] = []
        self.telemetry_records: List[dict] = []
        
        self.surrogate = None

    def _setup_optimizer(self) -> DynamicRFSurrogate:
        """Initializes the Dynamic RF surrogate model."""
        self.surrogate = DynamicRFSurrogate(
            extractor_name=self.extractor_name,
            window_size=self.window_size,
            n_base=self.n_base,
            n_min=self.n_min,
            n_max=self.n_max,
            gamma=self.gamma,
            depth_base=self.depth_base,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            beta=self.beta,
            extractor_kwargs=self.extractor_kwargs,
            rf_kwargs=self.rf_kwargs
        )
        return self.surrogate

    def convert_configspace(self, configspace: ConfigurationSpace) -> SearchSpace:
        return configspace

    def convert_to_trial(self, config: Configuration) -> TrialInfo:
        return TrialInfo(config=config)

    def get_current_incumbent(self) -> Incumbent:
        if not self.history:
            return None
        return min(self.history, key=lambda x: x[1].cost)

    def ask(self) -> TrialInfo:
        """
        Asks the optimizer for the next configuration to evaluate.
        Uses LCB acquisition function over a candidate pool.
        """
        # Ensure setup is complete
        if self.surrogate is None:
            self.setup_optimizer()
            
        if len(self.history) < self.n_init:
            # Warmstart / Initial random sampling
            config = self.configspace.sample_configuration()
            return self.convert_to_trial(config)
            
        # Bayesian Optimization phase
        # Sample candidate pool (5000 candidates)
        n_candidates = 5000
        candidates = [self.configspace.sample_configuration() for _ in range(n_candidates)]
        
        # Convert candidates to numpy array representation
        X_cand = np.array([cfg.get_array() for cfg in candidates])
        
        # Predict using surrogate (triggers updates inside sliding window adaptor)
        preds, epistemic_unc = self.surrogate.predict(X_cand)
        
        # Compute LCB acquisition function: mean - kappa * epistemic_unc
        lcb = preds - self.kappa * epistemic_unc
        
        # Select candidate that minimizes LCB
        best_idx = int(np.argmin(lcb))
        
        return self.convert_to_trial(candidates[best_idx])

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """
        Tells the optimizer about an evaluated configuration and cost.
        Re-fits the surrogate model and logs telemetry.
        """
        self.history.append((trial_info, trial_value))
        
        # Determine current surrogate parameter state for telemetry
        n_trees = self.n_base
        max_depth = self.depth_base
        if self.surrogate is not None:
            n_trees, max_depth = self.surrogate.adaptor.get_next_parameters()
            
        # Fit surrogate on updated training set
        if len(self.history) >= self.n_init:
            X_train = np.array([t[0].config.get_array() for t in self.history])
            y_train = np.array([t[1].cost for t in self.history])
            self.surrogate.fit(X_train, y_train)
            
        # Log telemetry record
        record = {
            "trial_idx": len(self.history) - 1,
            "config": dict(trial_info.config),
            "cost": float(trial_value.cost),
            "surrogate_n_estimators": n_trees,
            "surrogate_max_depth": max_depth,
            "virtual_time": float(trial_value.virtual_time)
        }
        self.telemetry_records.append(record)
        
        # Persist telemetry to disk
        self._write_telemetry()

    def _write_telemetry(self) -> None:
        """Saves current telemetry records to a JSON file."""
        if not self.telemetry_path:
            return
        data = {
            "task_name": self.task.name,
            "extractor_name": self.extractor_name,
            "kappa": self.kappa,
            "trials": self.telemetry_records
        }
        with open(self.telemetry_path, "w") as f:
            json.dump(data, f, indent=4)
