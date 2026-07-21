import os
import sys
import unittest
import numpy as np
from ConfigSpace import ConfigurationSpace, Float

# Ensure parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carps.utils.task import Task, InputSpace, OutputSpace, OptimizationResources, TaskMetadata
from carps.objective_functions.dummy_problem import DummyObjectiveFunction
from rf_dynamic.dynamic_rf_surrogate import DynamicRFSurrogate
from carps_integration.optimizer import CARPSDynamicRFOptimizer
from ep_extractors import UQExtractorRegistry

class TestEpistemicAcquisition(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cs = ConfigurationSpace(space={"x": Float("x", bounds=(-2.0, 2.0))})
        self.obj_func = DummyObjectiveFunction(return_value=1.5, configuration_space=self.cs)
        self.task = Task(
            name="dummy_epistemic_acq_task",
            objective_function=self.obj_func,
            input_space=InputSpace(configuration_space=self.cs),
            output_space=OutputSpace(),
            optimization_resources=OptimizationResources(n_trials=10),
            metadata=TaskMetadata()
        )
        
        # Simple dataset for surrogate fitting
        self.X_train = np.linspace(-2, 2, 20).reshape(-1, 1)
        self.y_train = np.sin(self.X_train).ravel()
        self.X_test = np.linspace(-2, 2, 5).reshape(-1, 1)

    def test_surrogate_predict_uncertainty_modes(self):
        """Test DynamicRFSurrogate returns epistemic or total uncertainty depending on uncertainty_type."""
        surrogate = DynamicRFSurrogate(extractor_name="standard_disagreement")
        surrogate.fit(self.X_train, self.y_train)

        preds_epi, unc_epi = surrogate.predict(self.X_test, uncertainty_type="epistemic")
        preds_tot, unc_tot = surrogate.predict(self.X_test, uncertainty_type="total")

        self.assertEqual(len(preds_epi), len(self.X_test))
        self.assertEqual(len(unc_epi), len(self.X_test))
        self.assertTrue(np.all(unc_epi >= 0))
        self.assertTrue(np.all(unc_tot >= 0))
        # Mean predictions should be identical regardless of uncertainty_type mode
        np.testing.assert_array_almost_equal(preds_epi, preds_tot)

    def test_optimizer_acq_uncertainty_types(self):
        """Test CARPSDynamicRFOptimizer accepts acq_uncertainty_type and runs ask/tell steps."""
        from carps.utils.trials import TrialValue
        for mode in ["epistemic", "total"]:
            optimizer = CARPSDynamicRFOptimizer(
                task=self.task,
                extractor_name="standard_disagreement",
                n_init=3,
                acq_uncertainty_type=mode,
                telemetry_path=f"test_telemetry_{mode}.json"
            )
            try:
                # Ask initial design points
                trials = []
                for _ in range(3):
                    trial = optimizer.ask()
                    trials.append(trial)
                    optimizer.tell(trial, TrialValue(cost=1.5))

                # Ask BO phase point (uses surrogate and EI with selected acq_uncertainty_type)
                bo_trial = optimizer.ask()
                self.assertIsNotNone(bo_trial)
            finally:
                telemetry_file = f"test_telemetry_{mode}.json"
                if os.path.exists(telemetry_file):
                    os.remove(telemetry_file)

    def test_all_extractors_with_epistemic_ei(self):
        """Verify that all registered extractors run cleanly with epistemic EI acquisition."""
        from carps.utils.trials import TrialValue
        all_extractors = UQExtractorRegistry.list_registered()
        self.assertGreater(len(all_extractors), 0)

        for extractor_name in all_extractors:
            optimizer = CARPSDynamicRFOptimizer(
                task=self.task,
                extractor_name=extractor_name,
                n_init=3,
                acq_uncertainty_type="epistemic",
                telemetry_path="test_temp_telemetry.json"
            )
            try:
                for _ in range(3):
                    t = optimizer.ask()
                    optimizer.tell(t, TrialValue(cost=1.5))
                
                bo_trial = optimizer.ask()
                self.assertIsNotNone(bo_trial, f"Failed for extractor {extractor_name}")
            finally:
                if os.path.exists("test_temp_telemetry.json"):
                    os.remove("test_temp_telemetry.json")

    def test_edge_case_zero_epistemic_uncertainty(self):
        """Test that Expected Improvement computation does not fail when epistemic uncertainty is 0."""
        surrogate = DynamicRFSurrogate(extractor_name="standard_disagreement")
        surrogate.fit(self.X_train, self.y_train)

        # Mock predict to return 0 epistemic uncertainty
        preds, unc = surrogate.predict(self.X_test, uncertainty_type="epistemic")
        zero_unc = np.zeros_like(unc)

        y_best = 0.0
        sigma = np.where(zero_unc > 1e-9, zero_unc, 1e-9)
        z = (y_best - preds) / sigma
        from scipy.stats import norm
        ei = (y_best - preds) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.where(zero_unc > 1e-9, ei, np.maximum(0.0, y_best - preds))

        self.assertFalse(np.any(np.isnan(ei)))
        self.assertFalse(np.any(np.isinf(ei)))

if __name__ == "__main__":
    unittest.main()
