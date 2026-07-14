import os
import sys
import unittest
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry

class TestUQRegistry(unittest.TestCase):
    def setUp(self):
        # Clear registry if populated to prevent interference between tests
        pass

    def test_base_class_is_abstract(self):
        # Verifies BaseEpistemicExtractor cannot be instantiated directly
        with self.assertRaises(TypeError):
            BaseEpistemicExtractor(None)

    def test_registry_registration_and_retrieval(self):
        # Define a mock extractor
        @UQExtractorRegistry.register("mock_extractor_for_test")
        class MockExtractor(BaseEpistemicExtractor):
            def __init__(self, model, some_param=10):
                super().__init__(model)
                self.some_param = some_param

            def fit(self, X_train, y_train):
                pass

            def extract_epistemic_signal(self, X):
                return np.zeros(len(X))

        # Retrieve and check
        self.assertIn("mock_extractor_for_test", UQExtractorRegistry.list_registered())
        
        # Instantiate
        mock_model = "fake_rf_model"
        extractor = UQExtractorRegistry.get("mock_extractor_for_test", mock_model, some_param=42)
        self.assertIsInstance(extractor, MockExtractor)
        self.assertEqual(extractor.model, mock_model)
        self.assertEqual(extractor.some_param, 42)

    def test_registry_duplicate_registration_error(self):
        # Register first
        @UQExtractorRegistry.register("duplicate_mock_test")
        class MockExtractor1(BaseEpistemicExtractor):
            def fit(self, X_train, y_train): pass
            def extract_epistemic_signal(self, X): return np.zeros(len(X))

        # Trying to register duplicate key should raise ValueError
        with self.assertRaises(ValueError):
            @UQExtractorRegistry.register("duplicate_mock_test")
            class MockExtractor2(BaseEpistemicExtractor):
                def fit(self, X_train, y_train): pass
                def extract_epistemic_signal(self, X): return np.zeros(len(X))

    def test_registry_invalid_lookup_error(self):
        # Fetching non-existent key should raise KeyError
        with self.assertRaises(KeyError):
            UQExtractorRegistry.get("non_existent_extractor", None)

if __name__ == "__main__":
    unittest.main()
