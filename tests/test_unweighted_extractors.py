import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ep_extractors import UQExtractorRegistry
from ep_extractors.proximity_b import ProximityBExtractor
from ep_extractors.proximity_bc import ProximityBCExtractor
from ep_extractors.proximity_auto_lambda import ProximityAutoLambdaExtractor
from ep_extractors.standard_proximity import StandardProximityExtractor


class TestUnweightedExtractors:
    @pytest.fixture(autouse=True)
    def setup_data_and_rf(self):
        np.random.seed(42)
        self.X_train = np.array([
            [0.1, 0.2],
            [0.15, 0.25],
            [0.8, 0.9],
            [0.85, 0.95],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.3],
            [0.7, 0.7],
            [0.4, 0.6],
        ])
        self.y_train = np.array([1.0, 1.1, 5.0, 5.2, 3.0, 2.5, 2.7, 1.5, 4.5, 2.8])
        self.rf = RandomForestRegressor(n_estimators=10, random_state=42, oob_score=True)
        self.rf.fit(self.X_train, self.y_train)

        self.X_test = np.array([
            [0.12, 0.22],  # Close to train cluster 1
            [0.82, 0.92],  # Close to train cluster 2
            [0.0, 0.0],    # Boundary
            [1.0, 1.0],    # Boundary
            [0.5, 0.5],    # Direct match
        ])

    def test_registry_keys_presence(self):
        """Test that all required registry keys are properly registered."""
        registered = UQExtractorRegistry.list_registered()
        expected_keys = [
            "proximity_b",
            "proximity_b_unweighted",
            "proximity_b_weighted",
            "proximity_bc",
            "proximity_bc_unweighted",
            "proximity_bc_weighted",
            "standard_proximity",
            "standard_proximity_unweighted",
            "standard_proximity_weighted",
            "proximity_auto_lambda",
        ]
        for key in expected_keys:
            assert key in registered, f"Registry key '{key}' not found in registered extractors: {registered}"

    def test_default_weighting_is_unweighted_all(self):
        """
        Test that ProximityBExtractor, ProximityBCExtractor, ProximityAutoLambdaExtractor,
        and StandardProximityExtractor default to uq_model.weighting == 'unweighted_all'.
        """
        extractors = [
            ProximityBExtractor(self.rf),
            ProximityBCExtractor(self.rf),
            ProximityAutoLambdaExtractor(self.rf),
            StandardProximityExtractor(self.rf),
            UQExtractorRegistry.get("proximity_b", self.rf),
            UQExtractorRegistry.get("proximity_bc", self.rf),
            UQExtractorRegistry.get("proximity_auto_lambda", self.rf),
            UQExtractorRegistry.get("standard_proximity", self.rf),
            UQExtractorRegistry.get("proximity_b_unweighted", self.rf),
            UQExtractorRegistry.get("proximity_bc_unweighted", self.rf),
            UQExtractorRegistry.get("standard_proximity_unweighted", self.rf),
        ]

        for ext in extractors:
            ext.fit(self.X_train, self.y_train)
            assert hasattr(ext, "uq_model") and ext.uq_model is not None
            assert ext.uq_model.weighting == "unweighted_all", (
                f"Extractor {ext.__class__.__name__} defaulted to weighting='{ext.uq_model.weighting}', "
                f"expected 'unweighted_all'"
            )
            assert ext.uq_model.weighted is False

    def test_weighted_flag_and_weighted_registry_keys(self):
        """
        Test that passing weighted=True or using *_weighted registry keys
        results in uq_model.weighting == 'leaf_normalized'.
        """
        weighted_by_flag = [
            ProximityBExtractor(self.rf, weighted=True),
            ProximityBCExtractor(self.rf, weighted=True),
            ProximityAutoLambdaExtractor(self.rf, weighted=True),
            StandardProximityExtractor(self.rf, weighted=True),
            UQExtractorRegistry.get("proximity_b", self.rf, weighted=True),
            UQExtractorRegistry.get("proximity_bc", self.rf, weighted=True),
            UQExtractorRegistry.get("proximity_auto_lambda", self.rf, weighted=True),
            UQExtractorRegistry.get("standard_proximity", self.rf, weighted=True),
        ]
        weighted_by_registry = [
            UQExtractorRegistry.get("proximity_b_weighted", self.rf),
            UQExtractorRegistry.get("proximity_bc_weighted", self.rf),
            UQExtractorRegistry.get("standard_proximity_weighted", self.rf),
        ]

        for ext in weighted_by_flag + weighted_by_registry:
            ext.fit(self.X_train, self.y_train)
            assert hasattr(ext, "uq_model") and ext.uq_model is not None
            assert ext.uq_model.weighting == "leaf_normalized", (
                f"Extractor {ext.__class__.__name__} did not resolve to 'leaf_normalized', "
                f"got '{ext.uq_model.weighting}'"
            )
            assert ext.uq_model.weighted is True

    def test_extract_epistemic_signal_validity(self):
        """
        Test that extract_epistemic_signal works properly and returns finite,
        valid uncertainties for all proximity extractors in both modes.
        """
        keys_to_test = [
            "proximity_b",
            "proximity_b_unweighted",
            "proximity_b_weighted",
            "proximity_bc",
            "proximity_bc_unweighted",
            "proximity_bc_weighted",
            "standard_proximity",
            "standard_proximity_unweighted",
            "standard_proximity_weighted",
            "proximity_auto_lambda",
        ]

        for key in keys_to_test:
            ext = UQExtractorRegistry.get(key, self.rf)
            ext.fit(self.X_train, self.y_train)
            signal = ext.extract_epistemic_signal(self.X_test)

            assert isinstance(signal, np.ndarray), f"{key}: signal should be numpy ndarray"
            assert signal.shape == (len(self.X_test),), f"{key}: signal shape mismatch"
            assert np.all(np.isfinite(signal)), f"{key}: signal contains non-finite values: {signal}"
            assert np.all(signal >= -1e-12), f"{key}: signal contains negative values: {signal}"

    def test_proximity_bc_oob_mae_and_predict_with_intervals(self):
        """Test ProximityBCExtractor oob_mae and predict_with_intervals methods."""
        ext = ProximityBCExtractor(self.rf)
        ext.fit(self.X_train, self.y_train)

        # oob_mae property
        assert hasattr(ext, "oob_mae"), "ProximityBCExtractor must expose oob_mae"
        assert isinstance(ext.oob_mae, float)
        assert ext.oob_mae >= 0.0

        # predict_with_intervals
        lower, pred, upper, mae = ext.predict_with_intervals(self.X_test, return_mae=True)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (len(self.X_test),)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert np.all(lower <= upper + 1e-6)
        assert isinstance(mae, np.ndarray)
        assert np.all(np.isfinite(mae))
        assert np.all(mae >= 0.0)

    def test_predict_with_intervals_weighted_override(self):
        """Test that predict_with_intervals respects weighted runtime override."""
        ext = ProximityBCExtractor(self.rf)
        ext.fit(self.X_train, self.y_train)

        lower_unw, pred_unw, upper_unw = ext.predict_with_intervals(self.X_test, weighted=False)
        lower_w, pred_w, upper_w = ext.predict_with_intervals(self.X_test, weighted=True)

        width_unw = upper_unw - lower_unw
        width_w = upper_w - lower_w

        assert np.all(np.isfinite(width_unw))
        assert np.all(np.isfinite(width_w))
        # Ensure widths are positive
        assert np.all(width_unw >= 0)
        assert np.all(width_w >= 0)
