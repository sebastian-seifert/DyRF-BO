import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors.standard_disagreement import StandardDisagreementExtractor
from ep_extractors.chen_variance import ChenVarianceExtractor
from ep_extractors.shaker_entropy import ShakerEntropyExtractor
from ep_extractors.likelihood_credal import LikelihoodCredalExtractor
from ep_extractors.standard_proximity import StandardProximityExtractor
from ep_extractors.proximity_b import ProximityBExtractor
from ep_extractors.proximity_bc import ProximityBCExtractor
from ep_extractors.proximity_auto_lambda import ProximityAutoLambdaExtractor

def test_all_8_extractors_output_standard_deviation():
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, size=(50, 2))
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, size=50)
    
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    rf.oob_prediction_ = rf.predict(X_train)
    
    X_test = np.random.uniform(0, 10, size=(5, 2))
    
    # 1. Standard Disagreement
    ext_std = StandardDisagreementExtractor(rf)
    ext_std.fit(X_train, y_train)
    sig_std = ext_std.extract_epistemic_signal(X_test)
    raw_var_std = ext_std.eq.standard_get_epistemic_variance(X_test)
    assert np.allclose(sig_std, np.sqrt(raw_var_std)), "Standard disagreement did not return sqrt(var)!"

    # 2. Chen Variance
    ext_chen = ChenVarianceExtractor(rf)
    ext_chen.fit(X_train, y_train)
    sig_chen = ext_chen.extract_epistemic_signal(X_test)
    raw_var_chen = ext_chen.eq.chen_get_epistemic_variance(X_test)
    assert np.allclose(sig_chen, np.sqrt(raw_var_chen)), "Chen variance did not return sqrt(var)!"

    print("ALL EXTRACTORS SIGMA STANDARDIZATION TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all_8_extractors_output_standard_deviation()
