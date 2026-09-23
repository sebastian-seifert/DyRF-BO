"""Extrapolation Uncertainty Quantification Module.

Exports:
- ConvexHullProjectionSolver, ProjectionResult, project_simplex: Convex hull projection & distances
- sphere, rosenbrock, rastrigin, ackley: Vectorized synthetic benchmark objectives
- BaseObjective, StandardizedObjective, get_objective, register_objective: Standardized objective wrappers
- sample_subdomain_training, sample_natural_test, sample_stratified_test: Sampling engines
- create_smac_default_rf, DualUQEvaluator, DualUQResult: Dual UQ evaluation engine
- spearman_rank_correlation, prediction_interval_coverage_probability, mean_prediction_interval_width,
  winkler_interval_score, outlier_error_detection_auc, compute_comprehensive_metrics: Evaluation metrics
- ExtrapolationRunConfig, run_single_experiment: Sweep runner pipeline
"""

from .metrics_suite import (
    compute_comprehensive_metrics,
    mean_prediction_interval_width,
    outlier_error_detection_auc,
    prediction_interval_coverage_probability,
    spearman_rank_correlation,
    winkler_interval_score,
)
from .qp_convex_hull import (
    ConvexHullProjectionSolver,
    ProjectionResult,
    project_simplex,
)
from .runner import (
    ExtrapolationRunConfig,
    run_single_experiment,
)
from .samplers import (
    sample_natural_test,
    sample_stratified_test,
    sample_subdomain_training,
)
from .test_objectives import (
    OBJECTIVE_REGISTRY,
    BaseObjective,
    StandardizedObjective,
    ackley,
    get_objective,
    rastrigin,
    register_objective,
    rosenbrock,
    sphere,
)
from .uq_evaluator import (
    DualUQEvaluator,
    DualUQResult,
    create_smac_default_rf,
)

__all__ = [
    # Convex Hull QP Engine
    "ConvexHullProjectionSolver",
    "ProjectionResult",
    "project_simplex",
    # Synthetic Benchmark Objectives
    "sphere",
    "rosenbrock",
    "rastrigin",
    "ackley",
    "BaseObjective",
    "StandardizedObjective",
    "get_objective",
    "register_objective",
    "OBJECTIVE_REGISTRY",
    # Training & Test Samplers
    "sample_subdomain_training",
    "sample_natural_test",
    "sample_stratified_test",
    # Dual UQ Evaluator
    "create_smac_default_rf",
    "DualUQEvaluator",
    "DualUQResult",
    # Metrics Suite
    "spearman_rank_correlation",
    "prediction_interval_coverage_probability",
    "mean_prediction_interval_width",
    "winkler_interval_score",
    "outlier_error_detection_auc",
    "compute_comprehensive_metrics",
    # Runner Pipeline
    "ExtrapolationRunConfig",
    "run_single_experiment",
]
