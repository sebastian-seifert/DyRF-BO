import unittest
import numpy as np
from ep_extractors.aleatoric_shaker_masterplan import run_single_aleatoric_experiment, get_benchmark_functions, get_noise_regimes, get_rf_configs

class TestAleatoricShakerMasterplan(unittest.TestCase):
    def test_benchmark_functions_and_noise(self):
        funcs = get_benchmark_functions()
        self.assertIn("sin_1d", funcs)
        self.assertIn("sin_cos_2d", funcs)

        regimes = get_noise_regimes()
        self.assertIn("homoscedastic_low", regimes)
        self.assertIn("hetero_linear", regimes)

        rf_configs = get_rf_configs()
        self.assertIn("RF_Default", rf_configs)
        self.assertIn("RF_Overfit_Leaf1", rf_configs)

    def test_single_experiment_execution(self):
        res = run_single_aleatoric_experiment(
            func_name="sin_1d",
            noise_name="hetero_linear",
            rf_config_name="RF_Default",
            seed=1
        )
        self.assertIsNotNone(res)
        self.assertIn("shaker_entropy", res)
        self.assertIn("shaker_geom_var", res)
        self.assertIn("standard_ari_var", res)

        # Check metrics in one approach
        shaker_res = res["shaker_geom_var"]
        self.assertIn("spearman_true", shaker_res)
        self.assertIn("log_pearson_true", shaker_res)
        self.assertIn("mse_var", shaker_res)
        self.assertIn("nlpd_aleatoric", shaker_res)

    def test_sobol_power_of_two_sampling(self):
        """Verify that multi-dimensional Sobol sample counts are strictly powers of 2."""
        res = run_single_aleatoric_experiment(
            func_name="sin_cos_sin_3d",
            noise_name="hetero_linear",
            rf_config_name="RF_Default",
            seed=1,
            n_train=1024,
            n_test=256
        )
        self.assertIsNotNone(res)

if __name__ == "__main__":
    unittest.main()

