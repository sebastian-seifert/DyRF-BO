#!/usr/bin/env python3
"""Programmatically introspects all 9 approaches and returns their exact hyperparameter specifications."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors import UQExtractorRegistry

def get_all_approach_hyperparameters() -> dict:
    hp = {}

    # 1. Standard SMAC3 BO Baseline
    hp["smac3_bo"] = {
        "description": "Standard SMAC3 BO Facade using total variance for Expected Improvement",
        "facade_class": "smac.facade.HyperparameterOptimizationFacade",
        "surrogate_model": "RandomForestRegressor",
        "n_trees": 100,
        "acq_func": "EI",
        "acq_xi": 0.0,
        "uncertainty_type": "total_variance",
        "n_init": 10,
        "init_design": "Sobol / Default Random",
        "enable_adaptation": False
    }

    # 2. Standard Disagreement
    hp["standard_disagreement"] = {
        "description": "Variance of mean predictions across ensemble trees (pure epistemic uncertainty)",
        "formula": "Var_tree(mean_t(x)) = (1/B) * sum_t (mu_t(x) - mu_bar(x))^2",
        "n_trees": 100,
        "acq_func": "EI",
        "acq_xi": 0.0,
        "uncertainty_type": "epistemic",
        "n_init": 10,
        "enable_adaptation": False
    }

    # 3. Standard Proximity
    hp["standard_proximity"] = {
        "description": "K-Nearest Neighbor (K-NN) density-decay epistemic uncertainty scaling",
        "formula": "ep_std * (1.0 - exp(-lambda * d_k(x)))",
        "n_neighbors": 20,
        "topological_decay_lambda": 1.0,
        "metric": "euclidean",
        "acq_func": "EI",
        "acq_xi": 0.0,
        "uncertainty_type": "epistemic"
    }

    # 4. Proximity Method B
    hp["proximity_b"] = {
        "description": "Topological tree-path distance decay proximity (RF-GAP)",
        "formula": "U_topo(x) = 1.0 - exp(-lambda * d_topo(x, D_train))",
        "topological_decay_lambda": 1.0,
        "use_density_scaling": False,
        "acq_func": "EI",
        "acq_xi": 0.0,
        "uncertainty_type": "epistemic"
    }

    # 5. Proximity Method BC
    hp["proximity_bc"] = {
        "description": "Joint topological tree-path distance + leaf sample density scaling",
        "formula": "U_topo_density(x) = U_topo(x) * (N_leaf(x))^(-alpha)",
        "topological_decay_lambda": 1.0,
        "use_density_scaling": True,
        "density_scaling_alpha": 1.0,
        "acq_func": "EI",
        "acq_xi": 0.0,
        "uncertainty_type": "epistemic"
    }

    # 6. Proximity Auto Lambda
    hp["proximity_auto_lambda"] = {
        "description": "Automatic spatial density-adaptive decay rate scaling",
        "formula": "lambda_auto = min_lambda + (max_lambda - min_lambda) * (1.0 - local_density)",
        "n_neighbors": 20,
        "use_density_scaling": True,
        "density_scaling_alpha": 1.0,
        "topological_decay_lambda_grid": [0.5, 1.0, 5.0],
        "acq_func": "EI",
        "acq_xi": 0.0,
        "uncertainty_type": "epistemic"
    }

    # 7. Chen Variance
    hp["chen_variance"] = {
        "description": "Chen et al. analytical Random Forest epistemic vs aleatoric variance decomposition",
        "formula": "sigma_ep^2(x) = (1/B) sum_b (mu_b(x) - mu(x))^2 + (1/B) sum_b (1/n_b) s_b^2",
        "min_samples_leaf": 2,
        "acq_func": "EI",
        "acq_xi": 0.0,
        "uncertainty_type": "epistemic"
    }

    # 8. Shaker Entropy
    hp["shaker_entropy"] = {
        "description": "Shaker & Hüllermeier Mutual Information / GMM Target Entropy (100 tree components)",
        "formula": "MI(x) = H_total(x) - H_aleatoric(x) over 100-tree target GMM density",
        "n_components": 100,
        "num_samples": 10000,
        "n_grid": 128,
        "acq_func": "EI",
        "acq_xi": 0.0,
        "uncertainty_type": "epistemic"
    }

    # 9. Likelihood Credal
    hp["likelihood_credal"] = {
        "description": "Credal likelihood ratio epistemic uncertainty under numerical root-finding",
        "formula": "LR(x) = L(theta_0|x) / sup_theta L(theta|x)",
        "root_solver": "bisect",
        "integral_method": "trapz",
        "confidence_level": 0.95,
        "acq_func": "EI",
        "acq_xi": 0.0,
        "uncertainty_type": "epistemic"
    }

    return hp

if __name__ == "__main__":
    import json
    print(json.dumps(get_all_approach_hyperparameters(), indent=2))
