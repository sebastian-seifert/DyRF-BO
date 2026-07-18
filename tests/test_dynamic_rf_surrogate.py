import os
import sys
import unittest
import numpy as np
import warnings
from pathlib import Path

# Ensure parent directory is in path
sys.path.append(str(Path(__file__).parent.parent))

from rf_dynamic.dynamic_rf_surrogate import DynamicRFSurrogate

class TestDynamicRFSurrogate(unittest.TestCase):
    def setUp(self):
        # Generate dummy dataset
        self.X = np.random.rand(20, 2)
        self.y = np.random.rand(20)

    def test_proximity_extractor_enables_oob_score(self):
        # Initialize surrogate with a proximity extractor
        surrogate = DynamicRFSurrogate(extractor_name="standard_proximity")
        
        # We catch warnings to ensure no refitting warnings are emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            surrogate.fit(self.X, self.y)
            
            # Assert that oob_score=True was enabled on the RF model
            self.assertTrue(surrogate.model.oob_score, "oob_score should be True for standard_proximity")
            
            # Verify no refitting warning from GPU_Proximity_Regression_UQ was raised
            refit_warnings = [
                warn for warn in w 
                if issubclass(warn.category, UserWarning) 
                and "fitted without oob_score=True. Refitting" in str(warn.message)
            ]
            self.assertEqual(len(refit_warnings), 0, "Warning should not be raised because oob_score was pre-enabled")

    def test_non_proximity_extractor_leaves_oob_score_false(self):
        # Initialize surrogate with standard_disagreement
        surrogate = DynamicRFSurrogate(extractor_name="standard_disagreement")
        surrogate.fit(self.X, self.y)
        
        # oob_score should remain False (default)
        self.assertFalse(surrogate.model.oob_score, "oob_score should be False for standard_disagreement")

if __name__ == "__main__":
    unittest.main()
