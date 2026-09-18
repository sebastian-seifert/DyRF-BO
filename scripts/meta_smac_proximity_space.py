#!/usr/bin/env python3
"""ConfigSpace and Initial Design for Proximity LCB Meta-Optimization Layer.

Defines the meta-hyperparameter search space:
- k in [5, 30] (integer, inclusive, default=25)
- decay_lambda in [0.2, 2.0] (float, default=1.345)
- eps in [0.02, 0.20] (float, default=0.1678)

Provides an initial design that guarantees the previous best configuration
(incumbent found via Sobol random search) is evaluated first.
"""

from __future__ import annotations

from typing import List, Optional

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.scenario import Scenario


def create_proximity_meta_configspace(seed: int = 42) -> ConfigurationSpace:
    """Creates the ConfigurationSpace for tuning Proximity LCB acquisition."""
    cs = ConfigurationSpace(name="proximity_lcb_meta_space", seed=seed)

    k = Integer("k", bounds=(5, 30), default=25)
    decay_lambda = Float("decay_lambda", bounds=(0.2, 2.0), default=1.345)
    eps = Float("eps", bounds=(0.02, 0.20), default=0.1678)

    cs.add([k, decay_lambda, eps])
    return cs


class IncumbentFirstInitialDesign(SobolInitialDesign):
    """Initial design that forces the default/incumbent configuration to be evaluated first,

    followed by Sobol quasi-random configurations for space-filling exploration.
    """

    def select_configurations(self) -> List[Configuration]:
        """Returns the incumbent as the very first configuration, followed by Sobol samples."""
        configs = super().select_configurations()
        incumbent = self._configspace.get_default_configuration()
        filtered = [c for c in configs if c != incumbent]
        return [incumbent] + filtered


def build_initial_design_with_incumbent(
    scenario: Scenario,
    n_configs: Optional[int] = 10,
    seed: Optional[int] = None,
) -> IncumbentFirstInitialDesign:
    """Factory creating an IncumbentFirstInitialDesign bound to the given scenario."""
    return IncumbentFirstInitialDesign(
        scenario=scenario,
        n_configs=n_configs,
        seed=seed or scenario.seed,
    )
