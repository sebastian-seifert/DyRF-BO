#!/usr/bin/env python3
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class DataConfig:
    ndim: int = 1
    n_samples: int = 1200
    ood_type: str = "hypercube"  # "hypercube" or "manifold"
    id_split: float = 0.7
    gap_type: str = "empty"      # "empty" or "sparse"
    scaling_law: str = "linear"  # "linear", "fractional", "leaf"
    sparse_multiplier: int = 12
    seed: int = 1
    noise_std: float = 0.1
    target_normalization: bool = True

@dataclass
class RFConfig:
    name: str = "A"
    n_estimators: int = 100
    min_samples_leaf: int = 5
    min_samples_split: int = 2
    max_features: str = "sqrt"
    bootstrap: bool = True
    oob_score: bool = True

    @classmethod
    def from_preset(cls, name: str) -> "RFConfig":
        preset_map = {
            "A": {"n_estimators": 100, "min_samples_leaf": 5, "min_samples_split": 2, "max_features": "sqrt"},
            "B": {"n_estimators": 500, "min_samples_leaf": 10, "min_samples_split": 2, "max_features": "sqrt"},
            "C": {"n_estimators": 1000, "min_samples_leaf": 25, "min_samples_split": 2, "max_features": "sqrt"},
        }
        if name not in preset_map:
            raise ValueError(f"Unknown RF config preset '{name}'. Choose from 'A', 'B', 'C'.")
        params = preset_map[name]
        return cls(name=name, **params)

@dataclass
class ExtractorConfig:
    approaches: List[str] = field(default_factory=lambda: [
        "Standard", "Chen", "Shaker_GMM_Entropy", "Shaker_Likelihood_GL_Bisect",
        "Shaker_Likelihood_GL_Newton", "Shaker_Likelihood_Trapz_Bisect", "Shaker_Likelihood_Trapz_Newton"
    ])
    credal_quadrature_points: int = 30
    credal_bisection_max_iter: int = 50
    credal_bisection_tol: float = 1e-5

@dataclass
class ProximityConfig:
    topological_decay_lambda: List[float] = field(default_factory=lambda: [0.5, 1.0, 5.0])
    k_neighbors: List[str] = field(default_factory=lambda: ["10", "20", "50", "auto"])
    use_density_scaling: bool = True
    density_scaling_alpha: List[float] = field(default_factory=lambda: [1.0, 5.0])
    methods: List[str] = field(default_factory=lambda: [
        "Proximity_Baseline", "Proximity_Method_B", "Proximity_Method_C", "Proximity_Method_B_C"
    ])

@dataclass
class AcquisitionConfig:
    name: str = "ei"  # "ei", "lcb", "pi"
    xi: float = 0.0
    beta: float = 1.96

@dataclass
class BenchmarkMasterConfig:
    data: DataConfig = field(default_factory=DataConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    extractors: ExtractorConfig = field(default_factory=ExtractorConfig)
    proximity: ProximityConfig = field(default_factory=ProximityConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkMasterConfig":
        data_cfg = DataConfig(**data.get("data", {}))
        rf_cfg = RFConfig(**data.get("rf", {}))
        ext_cfg = ExtractorConfig(**data.get("extractors", {}))
        prox_cfg = ProximityConfig(**data.get("proximity", {}))
        acq_cfg = AcquisitionConfig(**data.get("acquisition", {}))
        return cls(
            data=data_cfg,
            rf=rf_cfg,
            extractors=ext_cfg,
            proximity=prox_cfg,
            acquisition=acq_cfg
        )

    def generate_sweep_task_lines(self, func_name: str, output_dir: str = "results/sweep_raw") -> List[str]:
        """
        Generates CLI task command strings for sweeps by expanding hyperparameter grid combinations.
        """
        lines = []
        base_approaches = ",".join(self.extractors.approaches)
        prox_approaches = ",".join(self.proximity.methods)

        lambdas_str = ",".join(map(str, self.proximity.topological_decay_lambda))
        ks_str = ",".join(map(str, self.proximity.k_neighbors))
        alphas_str = ",".join(map(str, self.proximity.density_scaling_alpha))

        # Baseline task line
        base_line = (
            f"--function {func_name} --rf_config {self.rf.name} --seed {self.data.seed} "
            f"--gap_type {self.data.gap_type} --ood_type {self.data.ood_type} "
            f"--approaches {base_approaches} --output_dir {output_dir}"
        )
        lines.append(base_line)

        # Proximity task line with parameter lists
        prox_line = (
            f"--function {func_name} --rf_config {self.rf.name} --seed {self.data.seed} "
            f"--gap_type {self.data.gap_type} --ood_type {self.data.ood_type} "
            f"--topological_decay_lambda {lambdas_str} --k_neighbors {ks_str} "
            f"--density_scaling_alpha {alphas_str} --approaches {prox_approaches} "
            f"--output_dir {output_dir}"
        )
        if self.proximity.use_density_scaling:
            prox_line += " --use_density_scaling"
        lines.append(prox_line)

        return lines

