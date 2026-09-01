import os
import sys
import unittest
import numpy as np
from scipy.stats import norm

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAcquisitionFunctions(unittest.TestCase):
    def setUp(self):
        self.preds = np.array([1.0, 2.0, 0.5, 3.0])
        self.unc = np.array([0.5, 1.0, 0.1, 0.0])
        self.y_best = 1.0

    def test_expected_improvement_default(self):
        from carps_integration.acquisitions import ExpectedImprovement
        acq = ExpectedImprovement(xi=0.0)
        scores = acq.compute(self.preds, self.unc, self.y_best)
        self.assertEqual(len(scores), len(self.preds))
        # For candidate 0: preds=1.0, y_best=1.0, unc=0.5 -> diff=0, z=0 -> norm.pdf(0)*0.5 = 0.39894 * 0.5 = 0.19947
        self.assertAlmostEqual(scores[0], 0.5 * norm.pdf(0), places=4)
        # For candidate 3: unc=0.0 -> max(0, y_best - preds) = max(0, -2.0) = 0.0
        self.assertAlmostEqual(scores[3], 0.0, places=4)

    def test_expected_improvement_with_xi(self):
        from carps_integration.acquisitions import ExpectedImprovement
        acq_base = ExpectedImprovement(xi=0.0)
        acq_xi = ExpectedImprovement(xi=0.5)
        scores_base = acq_base.compute(self.preds, self.unc, self.y_best)
        scores_xi = acq_xi.compute(self.preds, self.unc, self.y_best)
        # Higher xi demands more improvement, lowering scores for moderate candidates
        self.assertLess(scores_xi[0], scores_base[0])

    def test_lower_confidence_bound(self):
        from carps_integration.acquisitions import LowerConfidenceBound
        acq = LowerConfidenceBound(beta=2.0)
        scores = acq.compute(self.preds, self.unc, self.y_best)
        # Score = -preds + beta * unc
        # Candidate 0: -1.0 + 2.0 * 0.5 = 0.0
        # Candidate 2: -0.5 + 2.0 * 0.1 = -0.3
        self.assertAlmostEqual(scores[0], 0.0, places=4)
        self.assertAlmostEqual(scores[2], -0.3, places=4)

    def test_probability_of_improvement(self):
        from carps_integration.acquisitions import ProbabilityOfImprovement
        acq = ProbabilityOfImprovement(xi=0.0)
        scores = acq.compute(self.preds, self.unc, self.y_best)
        # Score = norm.cdf((y_best - xi - preds) / unc)
        # Candidate 0: z = (1.0 - 1.0)/0.5 = 0 -> cdf(0) = 0.5
        self.assertAlmostEqual(scores[0], 0.5, places=4)

    def test_registry_factory(self):
        from carps_integration.acquisitions import (
            AcquisitionRegistry,
            ExpectedImprovement,
            LowerConfidenceBound,
            ProbabilityOfImprovement
        )
        ei_instance = AcquisitionRegistry.get("ei", xi=0.01)
        self.assertIsInstance(ei_instance, ExpectedImprovement)
        self.assertEqual(ei_instance.xi, 0.01)

        lcb_instance = AcquisitionRegistry.get("lcb", beta=2.5)
        self.assertIsInstance(lcb_instance, LowerConfidenceBound)
        self.assertEqual(lcb_instance.beta, 2.5)

        pi_instance = AcquisitionRegistry.get("pi", xi=0.05)
        self.assertIsInstance(pi_instance, ProbabilityOfImprovement)
        self.assertEqual(pi_instance.xi, 0.05)

    def test_registry_invalid_name(self):
        from carps_integration.acquisitions import AcquisitionRegistry
        with self.assertRaises(ValueError):
            AcquisitionRegistry.get("invalid_acq_name")

    def test_additive_lcb_translation_invariance(self):
        """Verify Additive LCB acquisition is invariant under arbitrary objective translation (shifts)."""
        from carps_integration.acquisitions import LowerConfidenceBound, AdditiveEpistemicAcquisition

        base_lcb = LowerConfidenceBound(beta=2.0)
        additive_lcb = AdditiveEpistemicAcquisition(base_acq=base_lcb)

        preds = np.array([1.0, 2.0, 0.5, 3.0])
        unc_tot = np.array([0.5, 1.0, 0.1, 0.2])
        u_ep = np.array([0.2, 0.8, 0.1, 0.5])
        y_best = 1.0
        beta_t = 1.5

        scores_orig = additive_lcb.compute_additive(preds, unc_tot, u_ep, y_best=y_best, beta_t=beta_t)

        # Shift by large positive and large negative offsets
        for offset in [+10000.0, -10000.0, +1e6, -1e6]:
            preds_shifted = preds + offset
            y_best_shifted = y_best + offset
            scores_shifted = additive_lcb.compute_additive(
                preds_shifted, unc_tot, u_ep, y_best=y_best_shifted, beta_t=beta_t
            )
            np.testing.assert_allclose(
                scores_shifted,
                scores_orig,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Additive LCB scores failed translation invariance for offset={offset}"
            )

if __name__ == "__main__":
    unittest.main()
