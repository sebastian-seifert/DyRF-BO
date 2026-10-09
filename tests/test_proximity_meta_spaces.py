"""Unit tests for ConfigSpace definitions and incumbent-first initial designs for Proximity A, B, AC, and BC."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smac import HyperparameterOptimizationFacade, Scenario
from scripts.meta_smac_proximity_space import (
    create_proximity_meta_configspace,
    build_initial_design_with_incumbent,
    IncumbentFirstInitialDesign,
)


class TestProximityMetaSpaces(unittest.TestCase):
    """Verifies ConfigSpace configurations and incumbent-first initial designs across proximity methods."""

    def test_proximity_a_backward_compatibility(self):
        """Method 'a' retains original bounds and defaults; default parameter method='a' works."""
        cs_default = create_proximity_meta_configspace()
        cs_explicit = create_proximity_meta_configspace(method="a")

        for cs in [cs_default, cs_explicit]:
            hps = dict(cs)
            self.assertEqual(set(hps.keys()), {"k", "decay_lambda", "eps"})

            # k: [5, 30], default 25
            self.assertEqual(hps["k"].lower, 5)
            self.assertEqual(hps["k"].upper, 30)
            self.assertEqual(hps["k"].default_value, 25)

            # decay_lambda: [0.2, 2.0], default 1.345
            self.assertAlmostEqual(hps["decay_lambda"].lower, 0.2, places=4)
            self.assertAlmostEqual(hps["decay_lambda"].upper, 2.0, places=4)
            self.assertAlmostEqual(hps["decay_lambda"].default_value, 1.345, places=4)

            # eps: [0.02, 0.20], default 0.1678 for method 'a'
            self.assertAlmostEqual(hps["eps"].lower, 0.02, places=4)
            self.assertAlmostEqual(hps["eps"].upper, 0.20, places=4)
            self.assertAlmostEqual(hps["eps"].default_value, 0.1678, places=4)

    def test_proximity_b_configspace(self):
        """Method 'b' includes eps in [0.02, 0.20] (default 0.16) and decay_lambda in [0.2, 2.0] (default 1.345)."""
        cs = create_proximity_meta_configspace(method="b")
        hps = dict(cs)

        self.assertEqual(set(hps.keys()), {"eps", "decay_lambda"})

        # eps: [0.02, 0.20], default 0.16
        self.assertAlmostEqual(hps["eps"].lower, 0.02, places=4)
        self.assertAlmostEqual(hps["eps"].upper, 0.20, places=4)
        self.assertAlmostEqual(hps["eps"].default_value, 0.16, places=4)

        # decay_lambda: [0.2, 2.0], default 1.345
        self.assertAlmostEqual(hps["decay_lambda"].lower, 0.2, places=4)
        self.assertAlmostEqual(hps["decay_lambda"].upper, 2.0, places=4)
        self.assertAlmostEqual(hps["decay_lambda"].default_value, 1.345, places=4)

    def test_proximity_ac_configspace(self):
        """Method 'ac' includes k, decay_lambda, eps (default 0.16), and alpha in [0.1, 2.0] (default 1.0)."""
        cs = create_proximity_meta_configspace(method="ac")
        hps = dict(cs)

        self.assertEqual(set(hps.keys()), {"k", "decay_lambda", "eps", "alpha"})

        # k: [5, 30], default 25
        self.assertEqual(hps["k"].lower, 5)
        self.assertEqual(hps["k"].upper, 30)
        self.assertEqual(hps["k"].default_value, 25)

        # decay_lambda: [0.2, 2.0], default 1.345
        self.assertAlmostEqual(hps["decay_lambda"].lower, 0.2, places=4)
        self.assertAlmostEqual(hps["decay_lambda"].upper, 2.0, places=4)
        self.assertAlmostEqual(hps["decay_lambda"].default_value, 1.345, places=4)

        # eps: [0.02, 0.20], default 0.16
        self.assertAlmostEqual(hps["eps"].lower, 0.02, places=4)
        self.assertAlmostEqual(hps["eps"].upper, 0.20, places=4)
        self.assertAlmostEqual(hps["eps"].default_value, 0.16, places=4)

        # alpha: [0.1, 2.0], default 1.0
        self.assertAlmostEqual(hps["alpha"].lower, 0.1, places=4)
        self.assertAlmostEqual(hps["alpha"].upper, 2.0, places=4)
        self.assertAlmostEqual(hps["alpha"].default_value, 1.0, places=4)

    def test_proximity_bc_configspace(self):
        """Method 'bc' includes eps (default 0.16), decay_lambda, and alpha (default 1.0)."""
        cs = create_proximity_meta_configspace(method="bc")
        hps = dict(cs)

        self.assertEqual(set(hps.keys()), {"eps", "decay_lambda", "alpha"})

        # eps: [0.02, 0.20], default 0.16
        self.assertAlmostEqual(hps["eps"].lower, 0.02, places=4)
        self.assertAlmostEqual(hps["eps"].upper, 0.20, places=4)
        self.assertAlmostEqual(hps["eps"].default_value, 0.16, places=4)

        # decay_lambda: [0.2, 2.0], default 1.345
        self.assertAlmostEqual(hps["decay_lambda"].lower, 0.2, places=4)
        self.assertAlmostEqual(hps["decay_lambda"].upper, 2.0, places=4)
        self.assertAlmostEqual(hps["decay_lambda"].default_value, 1.345, places=4)

        # alpha: [0.1, 2.0], default 1.0
        self.assertAlmostEqual(hps["alpha"].lower, 0.1, places=4)
        self.assertAlmostEqual(hps["alpha"].upper, 2.0, places=4)
        self.assertAlmostEqual(hps["alpha"].default_value, 1.0, places=4)

    def test_invalid_method_raises_value_error(self):
        """Unknown proximity method names should raise ValueError."""
        with self.assertRaises(ValueError):
            create_proximity_meta_configspace(method="unknown_method")

    def test_initial_design_cap_length_and_incumbent_first_all_methods(self):
        """IncumbentFirstInitialDesign caps returned configs to len(configs) and trial 1 is incumbent."""
        for method in ["a", "b", "ac", "bc"]:
            cs = create_proximity_meta_configspace(method=method)
            default_cfg = cs.get_default_configuration()

            with tempfile.TemporaryDirectory() as tmpdir:
                scenario = Scenario(
                    configspace=cs,
                    n_trials=50,
                    deterministic=True,
                    output_directory=Path(tmpdir) / f"smac_{method}",
                )

                # Test capping to exact n_configs (e.g. 5)
                n_requested = 5
                initial_design = build_initial_design_with_incumbent(scenario, n_configs=n_requested)
                configs = initial_design.select_configurations()
                self.assertEqual(
                    len(configs),
                    n_requested,
                    f"Method {method}: expected exactly {n_requested} configs, got {len(configs)}",
                )
                self.assertEqual(
                    configs[0],
                    default_cfg,
                    f"Method {method}: first configuration must be incumbent",
                )

                # Test asking SMAC facade returns default configuration first
                smac_opt = HyperparameterOptimizationFacade(
                    scenario=scenario,
                    target_function=lambda cfg, seed=0: 0.0,
                    initial_design=initial_design,
                    overwrite=True,
                )
                trial_info = smac_opt.ask()
                self.assertEqual(
                    trial_info.config,
                    default_cfg,
                    f"Method {method}: first SMAC trial must be incumbent",
                )


if __name__ == "__main__":
    unittest.main()
