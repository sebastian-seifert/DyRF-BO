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
        min_samples_leaf_base: int = 2,
        min_samples_leaf_min: int = 1,
        min_samples_leaf_max: int = 15,
        alpha: float = 1.0,
        max_features_base: float = 0.5,
        max_features_min: float = 0.1,
        max_features_max: float = 0.8,
        eta: float = 0.5,
        extractor_kwargs: dict | None = None,
        rf_kwargs: dict | None = None,
        acq_uncertainty_type: str = "epistemic"
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
        self.acq_uncertainty_type = acq_uncertainty_type
        
        # Dynamic RF hyperparameters
        self.window_size = window_size
        self.min_samples_leaf_base = min_samples_leaf_base
        self.min_samples_leaf_min = min_samples_leaf_min
        self.min_samples_leaf_max = min_samples_leaf_max
        self.alpha = alpha
        self.max_features_base = max_features_base
        self.max_features_min = max_features_min
        self.max_features_max = max_features_max
        self.eta = eta
        self.extractor_kwargs = extractor_kwargs or {}
        self.rf_kwargs = rf_kwargs or {}
        
        self.history: List[Tuple[TrialInfo, TrialValue]] = []
        self.telemetry_records: List[dict] = []
        
        self.surrogate = None
        self.initial_design_configs: List[Configuration] = []

    def _setup_optimizer(self) -> DynamicRFSurrogate:
        """Initializes the Dynamic RF surrogate model and Sobol initial design."""
        self.surrogate = DynamicRFSurrogate(
            extractor_name=self.extractor_name,
            window_size=self.window_size,
            min_samples_leaf_base=self.min_samples_leaf_base,
            min_samples_leaf_min=self.min_samples_leaf_min,
            min_samples_leaf_max=self.min_samples_leaf_max,
            alpha=self.alpha,
            max_features_base=self.max_features_base,
            max_features_min=self.max_features_min,
            max_features_max=self.max_features_max,
            eta=self.eta,
            extractor_kwargs=self.extractor_kwargs,
            rf_kwargs=self.rf_kwargs
        )
        
        # Setup SobolInitialDesign matching SMAC3
        try:
            import copy
            from smac.initial_design.sobol_design import SobolInitialDesign
            from smac.scenario import Scenario

            seed = getattr(self.task, "seed", None)
            if seed is None and hasattr(self.task, "objective_function"):
                seed = getattr(self.task.objective_function, "seed", None)
            if seed is None:
                seed = 0
            
            n_trials = getattr(getattr(self.task, "optimization_resources", None), "n_trials", 50) or 50
            scenario = Scenario(configspace=copy.deepcopy(self.configspace), seed=seed, n_trials=n_trials)
            sobol = SobolInitialDesign(scenario=scenario, n_configs=self.n_init)
            self.initial_design_configs = sobol.select_configurations()
        except Exception as e:
            print(f"Warning: Failed to initialize SobolInitialDesign ({e}), falling back to ConfigSpace sampling.")
            self.initial_design_configs = []

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
        Uses Expected Improvement (EI) acquisition function over a candidate pool.
        """
        # Ensure setup is complete
        if self.surrogate is None:
            self.setup_optimizer()
            
        if len(self.history) < self.n_init:
            # Warmstart / Initial Sobol sampling matching SMAC3
            if len(self.history) < len(self.initial_design_configs):
                config = self.initial_design_configs[len(self.history)]
            else:
                config = self.configspace.sample_configuration()
            return self.convert_to_trial(config)
            
        # Bayesian Optimization phase
        # Sample candidate pool (5000 candidates)
        n_candidates = 5000
        candidates = [self.configspace.sample_configuration() for _ in range(n_candidates)]
        
        # Convert candidates to numpy array representation (imputing NaNs in hierarchical spaces)
        X_cand = np.array([cfg.get_array() for cfg in candidates])
        X_cand = np.nan_to_num(X_cand, nan=-1.0)
        
        # Check if all runs in history failed (inf cost)
        valid_costs = [t[1].cost for t in self.history if not np.isinf(t[1].cost)]
        if not valid_costs:
            best_idx = np.random.randint(len(candidates))
            return self.convert_to_trial(candidates[best_idx])

        # Predict using surrogate (triggers updates inside sliding window adaptor)
        preds, unc = self.surrogate.predict(X_cand, uncertainty_type=self.acq_uncertainty_type)
        
        # Compute Expected Improvement (minimization: we want to improve below y_best)
        from scipy.stats import norm
        y_best = min(valid_costs)
        
        # Avoid division by zero
        sigma = np.where(unc > 1e-9, unc, 1e-9)
        z = (y_best - preds) / sigma
        
        ei = (y_best - preds) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Handle cases where uncertainty is close to zero
        ei = np.where(unc > 1e-9, ei, np.maximum(0.0, y_best - preds))
        
        # Select candidate that maximizes Expected Improvement
        best_idx = int(np.argmax(ei))
        return self.convert_to_trial(candidates[best_idx])

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """
        Tells the optimizer about an evaluated configuration and cost.
        Re-fits the surrogate model and logs telemetry.
        """
        self.history.append((trial_info, trial_value))
        
        # Determine current surrogate parameter state for telemetry
        min_samples_leaf = self.min_samples_leaf_base
        max_features = self.max_features_base
        if self.surrogate is not None:
            min_samples_leaf, max_features = self.surrogate.adaptor.get_next_parameters()
            
        # Fit surrogate on updated training set
        if len(self.history) >= self.n_init:
            X_train = np.array([t[0].config.get_array() for t in self.history])
            X_train = np.nan_to_num(X_train, nan=-1.0)
            y_train = np.array([t[1].cost for t in self.history])
            
            # Replace inf cost in failed runs with max finite cost + penalty for surrogate fitting
            finite_y = y_train[np.isfinite(y_train)]
            penalty_val = (float(np.max(finite_y)) + 10.0) if len(finite_y) > 0 else 1e9
            y_train = np.nan_to_num(y_train, posinf=penalty_val, neginf=-penalty_val)
            
            self.surrogate.fit(X_train, y_train)

            
        # Log telemetry record
        record = {
            "trial_idx": len(self.history) - 1,
            "config": dict(trial_info.config),
            "cost": float(trial_value.cost),
            "surrogate_min_samples_leaf": min_samples_leaf,
            "surrogate_max_features": max_features,
            "virtual_time": float(trial_value.virtual_time)
        }
        self.telemetry_records.append(record)
        
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
