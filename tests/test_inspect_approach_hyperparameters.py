import unittest

from scripts.inspect_approach_hyperparameters import get_all_approach_hyperparameters

class TestInspectApproachHyperparameters(unittest.TestCase):
    def test_get_all_approach_hyperparameters(self):
        hp_dict = get_all_approach_hyperparameters()
        
        # Check all 9 approaches are present
        expected_approaches = [
            "smac3_bo",
            "standard_disagreement",
            "standard_proximity",
            "proximity_b",
            "proximity_bc",
            "proximity_auto_lambda",
            "chen_variance",
            "shaker_entropy",
            "likelihood_credal"
        ]
        self.assertEqual(len(hp_dict), 9)
        for app in expected_approaches:
            self.assertIn(app, hp_dict)
            self.assertIsInstance(hp_dict[app], dict)
            self.assertGreater(len(hp_dict[app]), 0)

        # Spot check specific parameters
        self.assertEqual(hp_dict["smac3_bo"]["acq_func"], "EI")
        self.assertEqual(hp_dict["standard_proximity"]["n_neighbors"], 20)
        self.assertEqual(hp_dict["proximity_auto_lambda"]["use_density_scaling"], True)

if __name__ == "__main__":
    unittest.main()
