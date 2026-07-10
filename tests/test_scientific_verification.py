import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock
import scipy.special

# ==============================================================================
# 1. ROBUST CUPY MOCKING FOR CPU-GPU PARITY VERIFICATION
# ==============================================================================
class gp_array(np.ndarray):
    """
    Subclass of np.ndarray that mimics CuPy's device arrays.
    Crucially implements the .get() method to return a standard NumPy array,
    enabling seamless execution of GPU/CuPy code paths on the CPU.
    """
    def get(self):
        return np.asarray(self)

def to_gp(arr):
    """Converts a standard NumPy array or array-like to a gp_array view."""
    if isinstance(arr, np.ndarray):
        return arr.view(gp_array)
    return arr

class MockRNG:
    """Mock RNG class that wraps NumPy's Generator to return gp_array objects."""
    def __init__(self, rng):
        self.rng = rng
    def integers(self, *args, **kwargs):
        return to_gp(self.rng.integers(*args, **kwargs))
    def randint(self, *args, **kwargs):
        if hasattr(self.rng, 'randint'):
            return to_gp(self.rng.randint(*args, **kwargs))
        return to_gp(self.rng.integers(*args, **kwargs))
    def standard_normal(self, *args, **kwargs):
        return to_gp(self.rng.standard_normal(*args, **kwargs))
    def normal(self, *args, **kwargs):
        return to_gp(self.rng.normal(*args, **kwargs))

class MockCuPyModule:
    """Mock CuPy module routing GPU operations to NumPy while mimicking CUDA traits."""
    def __init__(self):
        self.ndarray = gp_array
        self.float32 = np.float32
        self.pi = np.pi
        self.nan = np.nan
        
        # Mock CUDA submodules
        self.cuda = MagicMock()
        self.cuda.runtime.getDeviceCount.return_value = 1
        
        # Mock 20 GB free, 24 GB total VRAM for dynamic batch calculations
        device_mock = MagicMock()
        device_mock.mem_info = (20 * 1024**3, 24 * 1024**3)
        self.cuda.Device.return_value = device_mock
        
        self.cuda.Stream = MagicMock()
        
        # Mock RNG
        self.random = MagicMock()
        self.random.default_rng = lambda *args, **kwargs: MockRNG(np.random.default_rng(*args, **kwargs))
        self.random.RandomState = lambda *args, **kwargs: MockRNG(np.random.RandomState(*args, **kwargs))
        self.random.standard_normal = lambda *args, **kwargs: to_gp(np.random.standard_normal(*args, **kwargs))
        
        # Array creation and manipulation
        self.asarray = lambda a, *args, **kwargs: to_gp(np.asarray(a, *args, **kwargs))
        self.asnumpy = lambda a, *args, **kwargs: np.asarray(a, *args, **kwargs).view(np.ndarray)
        self.array = lambda a, *args, **kwargs: to_gp(np.array(a, *args, **kwargs))
        self.arange = lambda *args, **kwargs: to_gp(np.arange(*args, **kwargs))
        self.log = lambda *args, **kwargs: to_gp(np.log(*args, **kwargs))
        self.sqrt = lambda *args, **kwargs: to_gp(np.sqrt(*args, **kwargs))
        self.mean = lambda *args, **kwargs: to_gp(np.mean(*args, **kwargs))
        self.median = lambda *args, **kwargs: to_gp(np.median(*args, **kwargs))
        self.min = lambda *args, **kwargs: to_gp(np.min(*args, **kwargs))
        self.max = lambda *args, **kwargs: to_gp(np.max(*args, **kwargs))
        self.minimum = lambda *args, **kwargs: to_gp(np.minimum(*args, **kwargs))
        self.maximum = lambda *args, **kwargs: to_gp(np.maximum(*args, **kwargs))
        self.exp = lambda *args, **kwargs: to_gp(np.exp(*args, **kwargs))
        self.abs = lambda *args, **kwargs: to_gp(np.abs(*args, **kwargs))
        self.sum = lambda *args, **kwargs: to_gp(np.sum(*args, **kwargs))
        self.zeros = lambda *args, **kwargs: to_gp(np.zeros(*args, **kwargs))
        self.ones = lambda *args, **kwargs: to_gp(np.ones(*args, **kwargs))
        self.flip = lambda *args, **kwargs: to_gp(np.flip(*args, **kwargs))
        self.argsort = lambda *args, **kwargs: to_gp(np.argsort(*args, **kwargs))
        self.quantile = lambda *args, **kwargs: to_gp(np.quantile(*args, **kwargs))
        self.nanquantile = lambda *args, **kwargs: to_gp(np.nanquantile(*args, **kwargs))
        self.broadcast_to = lambda *args, **kwargs: to_gp(np.broadcast_to(*args, **kwargs))
        self.where = lambda *args, **kwargs: to_gp(np.where(*args, **kwargs))
        self.any = lambda *args, **kwargs: to_gp(np.any(*args, **kwargs))
        self.cumsum = lambda *args, **kwargs: to_gp(np.cumsum(*args, **kwargs))
        self.newaxis = np.newaxis
        
        self.get_default_memory_pool = MagicMock()
        
        self.cupyx = MagicMock()
        self.cupyx.scipy.special.logsumexp = scipy.special.logsumexp
        self.cupyx.scipy.special.erf = scipy.special.erf
        self.cupyx.scipy.special.log_ndtr = scipy.special.log_ndtr
        self.cupyx.empty_pinned = lambda shape, dtype=None, order='C': np.empty(shape, dtype=dtype, order=order)

# Register the mock cupy/cupyx packages in sys.modules
mock_cp = MockCuPyModule()
sys.modules['cupy'] = mock_cp
sys.modules['cupyx'] = mock_cp.cupyx
sys.modules['cupyx.scipy'] = mock_cp.cupyx.scipy
sys.modules['cupyx.scipy.special'] = mock_cp.cupyx.scipy.special

# ==============================================================================
# 2. IMPORT CODE UNDER TEST
# ==============================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from Epistemic_Quantifier import EpistemicQuantifier
from Credal_Regression_UQ import CredalRegressionUQ
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from metrics import (
    calculate_rejection_curve,
    calculate_aurc,
    calculate_oracle_rejection_curve,
    calculate_random_rejection_curve,
    calculate_naurc,
    calculate_jensen_shannon_divergence,
    calculate_mutual_information,
    calculate_aurc_exact
)


# ==============================================================================
# 3. SCIENTIFIC VERIFICATION TEST SUITE
# ==============================================================================
class TestScientificVerification(unittest.TestCase):
    def setUp(self):
        """Prepare common synthetic datasets and fitted estimators."""
        np.random.seed(42)
        
        # 1. Standard Regression Dataset
        self.X_train = np.random.uniform(-3.0, 3.0, size=(100, 2))
        self.y_train = np.sin(self.X_train[:, 0]) + np.cos(self.X_train[:, 1]) + np.random.normal(0, 0.1, size=100)
        self.X_test = np.random.uniform(-3.0, 3.0, size=(20, 2))
        
        # Fit standard RandomForest
        self.rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=3, oob_score=True, random_state=42)
        self.rf.fit(self.X_train, self.y_train)

    def assertAllClose(self, a, b, rtol=1e-5, atol=1e-8, msg=None):
        """Assert that two float arrays are equal within relative/absolute tolerances."""
        a = np.asarray(a)
        b = np.asarray(b)
        self.assertEqual(a.shape, b.shape, f"Shape mismatch: {a.shape} vs {b.shape}")
        max_diff = np.max(np.abs(a - b))
        self.assertTrue(
            np.allclose(a, b, rtol=rtol, atol=atol),
            msg=f"{msg or 'Arrays are not close'}. Max difference: {max_diff:.6e} (rtol={rtol}, atol={atol})"
        )

    # --------------------------------------------------------------------------
    # 1. FLOATING-POINT TOLERANCES AND NUMERICAL PARITY
    # --------------------------------------------------------------------------
    def test_floating_point_tolerances(self):
        """Verify public methods using appropriate tolerances instead of strict equality."""
        # Test that multiple calls with the same configuration yield identical values under tolerance
        credal = CredalRegressionUQ(self.rf, self.X_train, self.y_train)
        ep1, al1 = credal.compute_uq(self.X_test, backend="cpu", n_grid=32)
        ep2, al2 = credal.compute_uq(self.X_test, backend="cpu", n_grid=32)
        
        self.assertAllClose(ep1, ep2, rtol=1e-12, atol=1e-12, msg="Repeated Credal compute_uq calls differ")
        self.assertAllClose(al1, al2, rtol=1e-12, atol=1e-12, msg="Repeated Credal compute_uq calls differ")

        prox = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu")
        uq1 = prox.compute_uq(self.X_test, n_neighbors='auto')
        uq2 = prox.compute_uq(self.X_test, n_neighbors='auto')
        self.assertAllClose(uq1, uq2, rtol=1e-12, atol=1e-12, msg="Repeated Proximity compute_uq calls differ")

    # --------------------------------------------------------------------------
    # 2. CPU-GPU PARITY
    # --------------------------------------------------------------------------
    def test_cpu_gpu_parity(self):
        """Verify that CPU and GPU (mock CuPy) backends yield equivalent outputs within 1e-6."""
        # A. CredalRegressionUQ Parity
        credal_cpu = CredalRegressionUQ(self.rf, self.X_train, self.y_train)
        ep_cpu, al_cpu = credal_cpu.compute_uq(self.X_test, backend="cpu", n_grid=32)
        
        credal_gpu = CredalRegressionUQ(self.rf, self.X_train, self.y_train)
        ep_gpu, al_gpu = credal_gpu.compute_uq(self.X_test, backend="gpu", n_grid=32)
        
        self.assertAllClose(ep_cpu, ep_gpu, rtol=1e-6, atol=1e-6, msg="Credal UQ Epistemic CPU vs GPU mismatch")
        self.assertAllClose(al_cpu, al_gpu, rtol=1e-6, atol=1e-6, msg="Credal UQ Aleatoric CPU vs GPU mismatch")

        # B. GPUProximityRegressionUQ Parity
        prox_cpu = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu")
        uq_cpu = prox_cpu.compute_uq(self.X_test, n_neighbors='auto')
        
        prox_gpu = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="gpu")
        uq_gpu = prox_gpu.compute_uq(self.X_test, n_neighbors='auto')
        
        self.assertAllClose(uq_cpu, uq_gpu, rtol=1e-6, atol=1e-6, msg="Proximity UQ CPU vs GPU mismatch")

        # C. EpistemicQuantifier (Shaker CPU vs GPU Parity under identical RNG)
        eq_cpu = EpistemicQuantifier(self.rf, self.X_train, self.y_train)
        sh_cpu = eq_cpu.shaker_get_epistemic_entropy(self.X_test, num_samples=1000, random_state=42, backend="cpu")
        sh_gpu = eq_cpu.shaker_get_epistemic_entropy(self.X_test, num_samples=1000, random_state=42, backend="gpu")
        
        self.assertAllClose(sh_cpu, sh_gpu, rtol=1e-6, atol=1e-6, msg="Shaker UQ CPU vs GPU mismatch")

    # --------------------------------------------------------------------------
    # 3. NUMERICAL STABILITY AND UNDERFLOW/OVERFLOW PROTECTION
    # --------------------------------------------------------------------------
    def test_numerical_stability_extreme_inputs(self):
        """Test near-zero variances, singular inputs, and extremely large inputs."""
        # 1. Singular/Constant inputs (where leaf variances are exactly zero before min_var scaling)
        X_train_const = np.ones((50, 2))
        y_train_const = np.ones(50) * 10.0
        X_test_const = np.ones((10, 2))
        
        rf_const = RandomForestRegressor(n_estimators=5, min_samples_leaf=2, oob_score=True, random_state=42)
        rf_const.fit(X_train_const, y_train_const)
        
        # Verify bisection solver stability
        credal_const = CredalRegressionUQ(rf_const, X_train_const, y_train_const)
        ep_c, al_c = credal_const.compute_uq(X_test_const, n_grid=16, sup_solver="bisection")
        self.assertFalse(np.isnan(ep_c).any() or np.isinf(ep_c).any(), "NaN/Inf in bisection solver on singular input")
        self.assertFalse(np.isnan(al_c).any() or np.isinf(al_c).any(), "NaN/Inf in bisection solver on singular input")
        
        # Verify Newton-Raphson solver stability
        ep_n, al_n = credal_const.compute_uq(X_test_const, n_grid=16, sup_solver="newton")
        self.assertFalse(np.isnan(ep_n).any() or np.isinf(ep_n).any(), "NaN/Inf in Newton solver on singular input")
        self.assertFalse(np.isnan(al_n).any() or np.isinf(al_n).any(), "NaN/Inf in Newton solver on singular input")

        # Verify GPUProximity stability
        prox_const = GPUProximityRegressionUQ(rf_const, X_train_const, y_train_const, device="cpu")
        uq_const = prox_const.compute_uq(X_test_const)
        self.assertFalse(np.isnan(uq_const).any() or np.isinf(uq_const).any(), "NaN/Inf in proximity UQ on singular input")

        # Verify EpistemicQuantifier stability
        eq_const = EpistemicQuantifier(rf_const, X_train_const, y_train_const)
        sh_const = eq_const.shaker_get_epistemic_entropy(X_test_const, num_samples=100, random_state=42)
        self.assertFalse(np.isnan(sh_const).any() or np.isinf(sh_const).any(), "NaN/Inf in shaker on singular input")

        # 2. Extremely large values (to test overflow protection in exp/log and special functions)
        X_train_large = np.random.uniform(-3.0, 3.0, size=(50, 2))
        y_train_large = np.random.uniform(-1, 1, size=50) * 1e12  # Out-of-bounds target variance
        X_test_large = np.random.uniform(-3.0, 3.0, size=(10, 2))
        
        rf_large = RandomForestRegressor(n_estimators=5, min_samples_leaf=2, oob_score=True, random_state=42)
        rf_large.fit(X_train_large, y_train_large)
        
        credal_large = CredalRegressionUQ(rf_large, X_train_large, y_train_large)
        ep_l, al_l = credal_large.compute_uq(X_test_large, n_grid=16)
        self.assertFalse(np.isnan(ep_l).any() or np.isinf(ep_l).any(), "NaN/Inf in Credal on extremely large input")
        self.assertFalse(np.isnan(al_l).any() or np.isinf(al_l).any(), "NaN/Inf in Credal on extremely large input")

    # --------------------------------------------------------------------------
    # 4. MATHEMATICAL INVARIANTS
    # --------------------------------------------------------------------------
    def test_mathematical_invariants(self):
        """Verify non-negativity of estimates, monotonicity of rejection curve, and NAURC bounds."""
        # 1. Non-negativity of UQ estimates
        credal = CredalRegressionUQ(self.rf, self.X_train, self.y_train)
        ep, al = credal.compute_uq(self.X_test, n_grid=32)
        self.assertTrue((ep >= 0.0).all(), "Negative epistemic uncertainty found in CredalRegressionUQ")
        self.assertTrue((al >= 0.0).all(), "Negative aleatoric uncertainty found in CredalRegressionUQ")

        prox = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu")
        uq_p = prox.compute_uq(self.X_test)
        self.assertTrue((uq_p >= 0.0).all(), "Negative uncertainty found in GPUProximityRegressionUQ")

        eq = EpistemicQuantifier(self.rf, self.X_train, self.y_train)
        sh_ep = eq.shaker_get_epistemic_entropy(self.X_test, num_samples=100)
        self.assertTrue((sh_ep >= 0.0).all(), "Negative entropy found in Shaker")

        # 2. Monotonicity of the Oracle Rejection Curve
        # The mean error of remaining samples must decrease/stay same as rejection rate increases
        rejection_rates = np.linspace(0.0, 0.95, 20)
        y_pred = self.rf.predict(self.X_test)
        oracle_curve = calculate_oracle_rejection_curve(y_pred, self.y_train[:20], rejection_rates, loss_type="MSE")
        
        for i in range(len(oracle_curve) - 1):
            self.assertTrue(
                oracle_curve[i] >= oracle_curve[i+1] - 1e-12,
                f"Oracle rejection curve violated monotonicity: {oracle_curve[i]} < {oracle_curve[i+1]}"
            )

        # 3. NAURC Limits
        # Oracle NAURC must be exactly 0.0, and Random NAURC must be exactly 1.0
        random_curve = calculate_random_rejection_curve(y_pred, self.y_train[:20], rejection_rates, loss_type="MSE", n_shuffles=20, random_state=42)
        
        naurc_oracle = calculate_naurc(rejection_rates, oracle_curve, oracle_curve, random_curve)
        naurc_random = calculate_naurc(rejection_rates, random_curve, oracle_curve, random_curve)
        
        self.assertAlmostEqual(naurc_oracle, 0.0, places=12, msg="Oracle NAURC is not 0.0")
        self.assertAlmostEqual(naurc_random, 1.0, places=12, msg="Random NAURC is not 1.0")

        # 4. CPU-GPU (Mock CuPy) Parity and Random Limit for calculate_aurc_exact
        p_max = 0.90
        # NumPy inputs
        u_np = np.random.uniform(0, 1, size=20)
        y_pred_np = np.random.normal(0, 1, size=20)
        y_true_np = np.random.normal(0, 1, size=20)
        
        # gp_array inputs (mocking GPU arrays)
        u_gp = to_gp(u_np.copy())
        y_pred_gp = to_gp(y_pred_np.copy())
        y_true_gp = to_gp(y_true_np.copy())
        
        # Calculate exact AURC
        aurc_exact_np = calculate_aurc_exact(u_np, y_pred_np, y_true_np, p_max=p_max)
        aurc_exact_gp = calculate_aurc_exact(u_gp, y_pred_gp, y_true_gp, p_max=p_max)
        
        # Parity Check: output must be exactly equal regardless of NumPy/CuPy array types
        self.assertAlmostEqual(aurc_exact_np, aurc_exact_gp, places=12, msg="Exact AURC has CPU-GPU parity discrepancy under mock CuPy testing")
        
        # Verify the analytical random limit logic:
        # A. For constant errors, aurc_exact should equal p_max * overall_error for any uncertainty
        constant_errors_pred = np.ones(20) * 5.0
        constant_errors_true = np.zeros(20)
        overall_error_const = np.mean((constant_errors_pred - constant_errors_true)**2)
        
        aurc_const_np = calculate_aurc_exact(u_np, constant_errors_pred, constant_errors_true, p_max=p_max)
        self.assertAlmostEqual(aurc_const_np, p_max * overall_error_const, places=12, msg="Analytical random limit for constant errors is not mathematically sound")
        
        # B. Convergence of average over shuffled uncertainties to p_max * overall_error
        shuffled_aurcs = []
        overall_error = np.mean((y_pred_np - y_true_np)**2)
        expected_random = p_max * overall_error
        
        # Shuffle uncertainty 500 times to approximate random UQ expectation
        for seed in range(500):
            rng = np.random.default_rng(seed)
            u_shuffled = rng.permutation(u_np)
            shuffled_aurcs.append(calculate_aurc_exact(u_shuffled, y_pred_np, y_true_np, p_max=p_max))
            
        mean_shuffled_aurc = np.mean(shuffled_aurcs)
        # Check convergence within 0.05 (statistical fluctuation for small N=20)
        self.assertAlmostEqual(mean_shuffled_aurc, expected_random, delta=0.05, 
                               msg=f"Expected shuffled AURC average to converge to random baseline {expected_random:.4f}, but got {mean_shuffled_aurc:.4f}")


    # --------------------------------------------------------------------------
    # 5. BATCH SIZE INDEPENDENCE
    # --------------------------------------------------------------------------
    def test_batch_size_independence(self):
        """Verify that UQ estimations are independent of evaluated batch sizing."""
        # 1. CredalRegressionUQ
        credal = CredalRegressionUQ(self.rf, self.X_train, self.y_train)
        ep_full, al_full = credal.compute_uq(self.X_test, backend="cpu", batch_size=100, n_grid=32)
        ep_batched, al_batched = credal.compute_uq(self.X_test, backend="cpu", batch_size=5, n_grid=32)
        ep_single, al_single = credal.compute_uq(self.X_test, backend="cpu", batch_size=1, n_grid=32)
        
        self.assertAllClose(ep_full, ep_batched, rtol=1e-12, atol=1e-12, msg="Credal batch size dependency (full vs batched)")
        self.assertAllClose(ep_full, ep_single, rtol=1e-12, atol=1e-12, msg="Credal batch size dependency (full vs single)")
        self.assertAllClose(al_full, al_batched, rtol=1e-12, atol=1e-12, msg="Credal batch size dependency (full vs batched)")
        self.assertAllClose(al_full, al_single, rtol=1e-12, atol=1e-12, msg="Credal batch size dependency (full vs single)")

        # 2. GPUProximityRegressionUQ
        prox_full = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu", batch_size=100)
        uq_full = prox_full.compute_uq(self.X_test)
        
        prox_batched = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu", batch_size=5)
        uq_batched = prox_batched.compute_uq(self.X_test)
        
        prox_single = GPUProximityRegressionUQ(self.rf, self.X_train, self.y_train, device="cpu", batch_size=1)
        uq_single = prox_single.compute_uq(self.X_test)
        
        self.assertAllClose(uq_full, uq_batched, rtol=1e-12, atol=1e-12, msg="Proximity batch size dependency (full vs batched)")
        self.assertAllClose(uq_full, uq_single, rtol=1e-12, atol=1e-12, msg="Proximity batch size dependency (full vs single)")

        # 3. EpistemicQuantifier (Shaker GMM Entropy Monte Carlo Batch Independence)
        eq = EpistemicQuantifier(self.rf, self.X_train, self.y_train)
        
        class DeterministicBatchRNG:
            def __init__(self, n_samples, num_samples, n_trees, random_state):
                rng = np.random.default_rng(random_state)
                self.master_components = rng.integers(0, n_trees, size=(n_samples, num_samples))
                self.master_eps = rng.normal(0, 1, size=(n_samples, num_samples)).astype(np.float32)
                self.comp_counter = 0
                self.eps_counter = 0
                
            def integers(self, low, high, size):
                B = size[0]
                if self.comp_counter + B > len(self.master_components):
                    self.comp_counter = 0
                res = self.master_components[self.comp_counter : self.comp_counter + B]
                self.comp_counter += B
                return to_gp(res)
                
            def normal(self, loc, scale, size):
                B = size[0]
                if self.eps_counter + B > len(self.master_eps):
                    self.eps_counter = 0
                res = self.master_eps[self.eps_counter : self.eps_counter + B]
                self.eps_counter += B
                return to_gp(res)
                
            def standard_normal(self, size):
                B = size[0]
                if self.eps_counter + B > len(self.master_eps):
                    self.eps_counter = 0
                res = self.master_eps[self.eps_counter : self.eps_counter + B]
                self.eps_counter += B
                return to_gp(res)
                
        def make_det_rng(seed):
            return DeterministicBatchRNG(len(self.X_test), 1000, len(self.rf.estimators_), seed)
            
        eq._mc_make_cpu_rng = make_det_rng
        eq._mc_make_gpu_rng = make_det_rng
        
        sh_full = eq.shaker_get_epistemic_entropy(self.X_test, num_samples=1000, batch_size=100, random_state=42, backend="cpu")
        sh_batched = eq.shaker_get_epistemic_entropy(self.X_test, num_samples=1000, batch_size=5, random_state=42, backend="cpu")
        sh_single = eq.shaker_get_epistemic_entropy(self.X_test, num_samples=1000, batch_size=1, random_state=42, backend="cpu")
        
        self.assertAllClose(sh_full, sh_batched, rtol=1e-12, atol=1e-12, msg="Shaker batch size dependency (full vs batched)")
        self.assertAllClose(sh_full, sh_single, rtol=1e-12, atol=1e-12, msg="Shaker batch size dependency (full vs single)")

    # --------------------------------------------------------------------------
    # 6. REPRODUCIBILITY AND DETERMINISM
    # --------------------------------------------------------------------------
    def test_reproducibility_and_determinism(self):
        """Verify that locked random seeds yield identical Monte Carlo outputs in Shaker UQ."""
        eq = EpistemicQuantifier(self.rf, self.X_train, self.y_train)
        
        # Running twice with the same seed must produce identical results
        sh_seed42_a = eq.shaker_get_epistemic_entropy(self.X_test, num_samples=1000, random_state=42, backend="cpu")
        sh_seed42_b = eq.shaker_get_epistemic_entropy(self.X_test, num_samples=1000, random_state=42, backend="cpu")
        self.assertAllClose(sh_seed42_a, sh_seed42_b, rtol=1e-12, atol=1e-12, msg="Locked seeds yielded different results")
        
        # Running with a different seed must produce different results
        sh_seed43 = eq.shaker_get_epistemic_entropy(self.X_test, num_samples=1000, random_state=43, backend="cpu")
        diff = np.max(np.abs(sh_seed42_a - sh_seed43))
        self.assertGreater(diff, 1e-4, "Different seeds yielded identical results")

    # --------------------------------------------------------------------------
    # ADDITIONAL TESTS FOR METRIC BOUNDS
    # --------------------------------------------------------------------------
    def test_metric_bounds(self):
        """Verify mathematical bounds of other entropy/divergence metrics."""
        y_true_binary = (self.y_train[:len(self.X_test)] > np.median(self.y_train)).astype(int)
        
        # Test Jensen-Shannon Divergence bounds [0, 1]
        jsd = calculate_jensen_shannon_divergence(self.X_test[:, 0], y_true_binary)
        if not np.isnan(jsd):
            self.assertTrue(0.0 <= jsd <= 1.0, f"JSD out of bounds: {jsd}")
            
        # Test Normalized Mutual Information (Uncertainty Coefficient) bounds [0, 1]
        nmi = calculate_mutual_information(self.X_test[:, 0], y_true_binary)
        if not np.isnan(nmi):
            self.assertTrue(0.0 <= nmi <= 1.0, f"NMI out of bounds: {nmi}")

if __name__ == "__main__":
    unittest.main()
