import os
import sys
import warnings
import numpy as np

# Reconfigure stdout and stderr to UTF-8 to prevent encoding issues in cluster environments
if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
if sys.stderr is not None:
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

class GPUProximityRegressionUQ:
    def __init__(self, model, X_train, y_train, device="auto", batch_size="auto", use_density_scaling=False, density_scaling_alpha=1.0, topological_decay_lambda=None):
        """
        GPU-Accelerated Wrapper for Localized Uncertainty Quantification in Random Forests
        via Proximities (RF-FIRE / RF-GAP). Supports dynamic NumPy and CuPy backends.
        
        Args:
            model: A fitted or unfitted RandomForestRegressor.
            X_train: array-like of shape (n_samples, n_features), training inputs.
            y_train: array-like of shape (n_samples,), training targets.
            device: str, "cpu", "gpu" (or "cuda"), or "auto".
                "auto" enables CuPy if a GPU and CuPy are available.
            batch_size: int or 'auto', size of chunked batches for test point processing.
                If 'auto', dynamically determines the optimal batch size based on free VRAM.
            use_density_scaling: bool, if True, scales proximity uncertainty inversely with leaf density.
            density_scaling_alpha: float, power exponent for leaf density scaling.
            topological_decay_lambda: float or None, exponential decay factor lambda for topological walking.
        """
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.n_train = len(self.X_train)
        self.batch_size_param = batch_size
        self.use_density_scaling = use_density_scaling
        self.density_scaling_alpha = density_scaling_alpha
        self.topological_decay_lambda = topological_decay_lambda

        
        # Configure backend dynamically
        self._init_backend(device)
        
        # If the model is already fitted, we can run fit() to extract necessary parameters
        if hasattr(self.model, "estimators_"):
            self.fit()

    def _init_backend(self, device):
        """Initializes the backend (NumPy or CuPy) dynamically."""
        device = device.lower()
        self.using_gpu = False
        
        if device in ("gpu", "cuda"):
            if HAS_CUPY:
                try:
                    # Quick check to verify GPU visibility/accessibility
                    device_count = cp.cuda.runtime.getDeviceCount()
                    if device_count > 0:
                        self.using_gpu = True
                    else:
                        warnings.warn("CuPy is installed but no GPUs were detected. Falling back to CPU.")
                except Exception as e:
                    warnings.warn(f"Failed to initialize CUDA device via CuPy: {e}. Falling back to CPU.")
            else:
                warnings.warn("GPU backend requested but CuPy is not installed. Falling back to CPU.")
        elif device == "auto":
            if HAS_CUPY:
                try:
                    if cp.cuda.runtime.getDeviceCount() > 0:
                        self.using_gpu = True
                except Exception:
                    pass
        
        if self.using_gpu:
            self.xp = cp
            # Check for nanquantile support in the current CuPy version
            try:
                _ = cp.nanquantile(cp.array([1.0, cp.nan]), 0.5)
                self.nanquantile_supported = True
            except (AttributeError, NotImplementedError):
                self.nanquantile_supported = False
        else:
            self.xp = np
            self.nanquantile_supported = True

    def _get_dynamic_batch_size(self, n_test):
        """
        Dynamically computes the optimal batch size based on available GPU VRAM.
        If running on CPU, returns a cache-friendly batch size to prevent memory thrashing.
        """
        if not self.using_gpu:
            # For CPU, keep the memory footprint per batch around 50 million elements (~50 MB)
            # to fit nicely within standard CPU L3 caches and prevent major page faults.
            total_elements = self.n_train * self.n_estimators
            if total_elements <= 0:
                return 128
            opt_batch = 50_000_000 // total_elements
            return max(1, min(n_test, min(128, opt_batch)))
        
        try:
            # Query free VRAM in bytes
            free_mem, total_mem = cp.cuda.Device().mem_info
            
            # Conservative estimate of memory consumed per query sample in bytes:
            # Shape (1, n_train, n_estimators) of float32 (4 bytes) + bool match (1 byte)
            # plus intermediate buffers. Let's budget 10 bytes per (n_train * n_estimators)
            bytes_per_sample = 10 * self.n_train * self.n_estimators
            
            if bytes_per_sample <= 0:
                return 256
            
            # Reserve 20% of VRAM for safety (CUDA context, libraries, other variables)
            usable_vram = 0.8 * free_mem
            
            # Compute optimal batch size
            opt_batch = int(usable_vram // bytes_per_sample)
            
            # Bound the batch size to be between 1 and n_test
            return max(1, min(n_test, opt_batch))
        except Exception:
            return 256  # Safe fallback if querying GPU VRAM fails

    def fit(self):
        """
        Fits the underlying Random Forest model if not already fitted,
        extracts OOB residual statistics, and prepares internal structures.
        """
        import time
        debug_timing = os.environ.get("PROXIMITY_DEBUG") == "1"
        
        if debug_timing:
            if self.using_gpu:
                try:
                    cp.cuda.Device().synchronize()
                except Exception:
                    pass
            t_start = time.perf_counter()
            
        if not hasattr(self.model, "estimators_"):
            # Ensure oob_score=True is enabled to collect out-of-bag statistics
            self.model.set_params(oob_score=True)
            self.model.fit(self.X_train, self.y_train)
        elif not getattr(self.model, "oob_score", False):
            # Model was fitted but lacks OOB predictions; we refit it with oob_score=True enabled
            warnings.warn("The provided RandomForestRegressor was fitted without oob_score=True. Refitting to extract OOB stats.", UserWarning)
            self.model.set_params(oob_score=True)
            self.model.fit(self.X_train, self.y_train)
            
        if debug_timing:
            t_fit_model = time.perf_counter()
            print(f"[TIMING] Random Forest fit/check: {(t_fit_model - t_start)*1000:.2f} ms")
            
        self.estimators = self.model.estimators_
        self.n_estimators = len(self.estimators)
        
        # Extract out-of-bag predictions and compute residuals
        self.oob_prediction_ = self.model.oob_prediction_
        self.oob_residuals = self.y_train - self.oob_prediction_
        
        # Import underlying scikit-learn helper functions to reconstruct OOB indices and in-bag counts
        from sklearn.ensemble._forest import _generate_unsampled_indices, _generate_sample_indices
        
        self.oob_indices = np.zeros((self.n_train, self.n_estimators), dtype=np.int32)
        self.in_bag_counts = np.zeros((self.n_train, self.n_estimators), dtype=np.int32)
        
        for t, tree in enumerate(self.estimators):
            # 1. Unsampled indices = Out-of-Bag (OOB) samples
            oob_idx = _generate_unsampled_indices(tree.random_state, self.n_train, self.n_train)
            self.oob_indices[oob_idx, t] = 1
            
            # 2. Sampled indices = In-bag samples (with frequency counts)
            ib_idx = _generate_sample_indices(tree.random_state, self.n_train, self.n_train)
            idx, counts = np.unique(ib_idx, return_counts=True)
            self.in_bag_counts[idx, t] = counts
            
        self.in_bag_indices = 1 - self.oob_indices
        self.leaf_matrix_train = self.model.apply(self.X_train)
        
        # in_bag_leaves keeps the leaf ID for in-bag samples, and sets to 0 for OOB samples
        self.in_bag_leaves = self.in_bag_indices * self.leaf_matrix_train
        
        if debug_timing:
            t_indices = time.perf_counter()
            print(f"[TIMING] OOB/In-bag index reconstruction: {(t_indices - t_fit_model)*1000:.2f} ms")
            
        # Precompute scaled training weights to avoid 3D array broadcasting inside compute_uq loop
        self.train_weights = np.zeros((self.n_train, self.n_estimators), dtype=np.float32)
        self.leaf_sizes = []
        for t, tree in enumerate(self.estimators):
            node_count = tree.tree_.node_count
            leaf_sums = np.bincount(
                self.in_bag_leaves[:, t],
                weights=self.in_bag_counts[:, t],
                minlength=node_count
            )
            # Avoid division by zero for any nodes that have no in-bag samples (e.g. OOB padding at index 0)
            leaf_sums[leaf_sums == 0.0] = 1.0
            
            # Store leaf sizes in active backend
            self.leaf_sizes.append(self.xp.asarray(leaf_sums))
            
            train_leaves = self.leaf_matrix_train[:, t]
            self.train_weights[:, t] = (self.in_bag_indices[:, t] * self.in_bag_counts[:, t]) / leaf_sums[train_leaves]
            
        # Precompute baseline leaf size for the training set (average leaf size across trees)
        train_leaf_sizes = self.xp.zeros((self.n_train, self.n_estimators), dtype=self.xp.float32)
        for t in range(self.n_estimators):
            train_leaf_sizes[:, t] = self.leaf_sizes[t][self.xp.asarray(self.leaf_matrix_train[:, t])]
        self.train_avg_leaf_sizes = self.xp.mean(train_leaf_sizes, axis=1)
        self.N_baseline = float(self.xp.median(self.train_avg_leaf_sizes))

        if debug_timing:
            t_weights = time.perf_counter()
            print(f"[TIMING] Leaf weights & density precomputation: {(t_weights - t_indices)*1000:.2f} ms")
            
        # Transfer training structures to the active backend (numpy or cupy)
        self.oob_residuals_xp = self.xp.asarray(self.oob_residuals)
        self.in_bag_leaves_xp = self.xp.asarray(self.in_bag_leaves)
        self.train_weights_xp = self.xp.asarray(self.train_weights)
        self.in_bag_counts_xp = self.xp.asarray(self.in_bag_counts)

        
        if debug_timing:
            if self.using_gpu:
                try:
                    cp.cuda.Device().synchronize()
                except Exception:
                    pass
            t_transfer = time.perf_counter()
            print(f"[TIMING] Transfer structures to backend: {(t_transfer - t_weights)*1000:.2f} ms")
            print(f"[TIMING] Total fit: {(t_transfer - t_start)*1000:.2f} ms")

    def compute_uq(self, X_test, n_neighbors="auto", level=0.95, use_density_scaling=None):
        """
        Computes localized uncertainty quantification (interval width) for test query points.
        
        Args:
            X_test: array-like of shape (n_test_samples, n_features), query points.
            n_neighbors: int, 'auto', or 'all'. Number of nearest neighbors to consider.
            level: float, confidence level of the prediction interval.
            use_density_scaling: bool or None, if True, scales UQ inversely with density.
        """
        if use_density_scaling is None:
            use_density_scaling = getattr(self, "use_density_scaling", False)

        # Ensure model is fitted and structures are prepared
        if not hasattr(self, "estimators"):
            self.fit()
            
        X_test = np.asarray(X_test)
        n_test = len(X_test)
        
        # Validate and parse n_neighbors parameter to be semantically equivalent to CPU version
        if n_neighbors != "auto" and n_neighbors != "all":
            try:
                # Round floats to match original CPU behavior exactly
                if isinstance(n_neighbors, float) or (isinstance(n_neighbors, str) and "." in n_neighbors):
                    n_neighbors_val = int(round(float(n_neighbors)))
                    warnings.warn(f"n_neighbors value {n_neighbors} is a float/float-string. Rounding to integer: {n_neighbors_val}.", UserWarning)
                else:
                    n_neighbors_val = int(n_neighbors)
                
                if n_neighbors_val <= 0 or n_neighbors_val > self.n_train:
                    raise ValueError(f"n_neighbors must be between 1 and {self.n_train} (n_train).")
                n_neighbors = n_neighbors_val
            except (ValueError, TypeError):
                raise ValueError("n_neighbors must be a positive integer, 'auto', or 'all'.")
        
        # Prepare debug timing
        import time
        debug_timing = os.environ.get("PROXIMITY_DEBUG") == "1"
        
        if debug_timing:
            if self.using_gpu:
                try:
                    cp.cuda.Device().synchronize()
                except Exception:
                    pass
            t_start = time.perf_counter()
            
        # Apply the trees to test points to get leaf IDs
        leaf_matrix_test = self.model.apply(X_test)
        leaf_matrix_test_xp = self.xp.asarray(leaf_matrix_test)
        
        if debug_timing:
            if self.using_gpu:
                try:
                    cp.cuda.Device().synchronize()
                except Exception:
                    pass
            t_leaf = time.perf_counter()
            print(f"[TIMING] Test leaf ID extraction: {(t_leaf - t_start)*1000:.2f} ms")
            
        # Pre-allocate output arrays on the backend
        resid_lwr = self.xp.zeros(n_test, dtype=self.xp.float32)
        resid_upr = self.xp.zeros(n_test, dtype=self.xp.float32)
        
        alpha_lwr = (1.0 - level) / 2.0
        alpha_upr = 1.0 - alpha_lwr
        
        # Determine batch size dynamically if auto, else parse parameter
        if self.batch_size_param == "auto":
            batch_size = self._get_dynamic_batch_size(n_test)
        else:
            batch_size = int(self.batch_size_param)
            
        t_accum_total = 0.0
        t_quantile_total = 0.0
        
        # Pre-allocate walked density storage if using Method C
        if use_density_scaling and self.topological_decay_lambda is not None and self.topological_decay_lambda > 0.0:
            walked_densities = self.xp.zeros(n_test, dtype=self.xp.float32)
            
        # Process in batches to control GPU/CPU memory consumption
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            batch_len = end - start
            
            leaf_batch = leaf_matrix_test_xp[start:end, :]  # (batch_size, n_estimators)
            
            if debug_timing:
                if self.using_gpu:
                    try:
                        cp.cuda.Device().synchronize()
                    except Exception:
                        pass
                t0_accum = time.perf_counter()
                
            # Efficient tree-by-tree proximity accumulation to avoid allocating a massive 3D tensor
            # prox_batch has shape (batch_size, n_train)
            prox_batch = self.xp.zeros((batch_len, self.n_train), dtype=self.xp.float32)
            
            # Initialize dynamic density accumulator if Method C is active
            if use_density_scaling and self.topological_decay_lambda is not None and self.topological_decay_lambda > 0.0:
                density_batch = self.xp.zeros(batch_len, dtype=self.xp.float32)
                
            for t in range(self.n_estimators):
                if self.topological_decay_lambda is not None and self.topological_decay_lambda > 0.0:
                    # Compute topological leaf path distances
                    d_t = self.compute_tree_topological_distances(leaf_batch[:, t], self.in_bag_leaves_xp[:, t], t)
                    # Compute exponential decay kernel: e^(-lambda * d_t)
                    decay_t = self.xp.exp(-self.topological_decay_lambda * d_t)
                    # Accumulate walked proximity
                    prox_batch += decay_t * self.train_weights_xp[None, :, t]
                    
                    if use_density_scaling:
                        # Accumulate walked density sum (unnormalized density metric)
                        density_batch += self.xp.sum(decay_t * self.in_bag_counts_xp[None, :, t], axis=1)
                else:
                    # 2D comparison: (batch_size, 1) == (1, n_train) -> (batch_size, n_train)
                    # Matches if test sample leaf equals train sample leaf in tree t
                    matches_t = leaf_batch[:, t, None] == self.in_bag_leaves_xp[None, :, t]
                    # Accumulate the precomputed weights
                    prox_batch += matches_t * self.train_weights_xp[None, :, t]
                
            prox_batch /= self.n_estimators
            
            if use_density_scaling and self.topological_decay_lambda is not None and self.topological_decay_lambda > 0.0:
                walked_densities[start:end] = density_batch / self.n_estimators
            
            if debug_timing:
                if self.using_gpu:
                    try:
                        cp.cuda.Device().synchronize()
                    except Exception:
                        pass
                t1_accum = time.perf_counter()
                t_accum_total += (t1_accum - t0_accum)
                t0_quantile = time.perf_counter()
                
            if n_neighbors == "auto":
                if self.topological_decay_lambda is not None and self.topological_decay_lambda > 0.0:
                    # Method B: Topological Weighted Quantiles
                    resid_lwr[start:end] = self._compute_weighted_quantile(self.oob_residuals_xp, prox_batch, alpha_lwr)
                    resid_upr[start:end] = self._compute_weighted_quantile(self.oob_residuals_xp, prox_batch, alpha_upr)
                else:
                    # Mask out training samples with proximity < 1e-10 using xp.where instead of tiling
                    masked_residuals = self.xp.where(prox_batch >= 1e-10, self.oob_residuals_xp[None, :], self.xp.nan)
                    
                    # Perform nanquantile estimation
                    if self.using_gpu and not self.nanquantile_supported:
                        # Fall back to NumPy CPU for nanquantile if CuPy version lacks it
                        tiled_cpu = cp.asnumpy(masked_residuals)
                        lwr_cpu = np.nanquantile(tiled_cpu, alpha_lwr, axis=1)
                        upr_cpu = np.nanquantile(tiled_cpu, alpha_upr, axis=1)
                        resid_lwr[start:end] = cp.asarray(lwr_cpu)
                        resid_upr[start:end] = cp.asarray(upr_cpu)
                    else:
                        resid_lwr[start:end] = self.xp.nanquantile(masked_residuals, alpha_lwr, axis=1)
                        resid_upr[start:end] = self.xp.nanquantile(masked_residuals, alpha_upr, axis=1)
            else:
                k = self.n_train if n_neighbors == "all" else int(n_neighbors)
                
                # If k is less than self.n_train, we extract the top k proximate neighbors
                if k < self.n_train:
                    # Use argsort to match the reference implementation's tie-breaking behavior exactly
                    partition_idx = self.xp.flip(self.xp.argsort(prox_batch, axis=1), axis=1)[:, :k]
                    
                    # Extract corresponding residuals directly via indexing (fancy indexing)
                    k_residuals = self.oob_residuals_xp[partition_idx]
                else:
                    # If k matches self.n_train, sorting is redundant; use broadcasted residuals
                    k_residuals = self.xp.broadcast_to(self.oob_residuals_xp[None, :], (batch_len, self.n_train))
                
                resid_lwr[start:end] = self.xp.quantile(k_residuals, alpha_lwr, axis=1)
                resid_upr[start:end] = self.xp.quantile(k_residuals, alpha_upr, axis=1)
                
            if debug_timing:
                if self.using_gpu:
                    try:
                        cp.cuda.Device().synchronize()
                    except Exception:
                        pass
                t1_quantile = time.perf_counter()
                t_quantile_total += (t1_quantile - t0_quantile)
                
        if debug_timing:
            print(f"[TIMING] Proximity matrix accumulation: {t_accum_total*1000:.2f} ms")
            print(f"[TIMING] Neighbor sorting & quantiles: {t_quantile_total*1000:.2f} ms")
            print(f"[TIMING] Total compute_uq: {(time.perf_counter() - t_start)*1000:.2f} ms")
            
        # Calculate localized prediction interval width (uncertainty)
        uq = resid_upr - resid_lwr
        
        if use_density_scaling:
            if self.topological_decay_lambda is not None and self.topological_decay_lambda > 0.0:
                # Method C: Topological Density Scaling
                avg_test_leaf_sizes = walked_densities
            else:
                # Standard Density Scaling
                test_leaf_sizes = self.xp.zeros((n_test, self.n_estimators), dtype=self.xp.float32)
                for t in range(self.n_estimators):
                    test_leaf_sizes[:, t] = self.leaf_sizes[t][leaf_matrix_test_xp[:, t]]
                avg_test_leaf_sizes = self.xp.mean(test_leaf_sizes, axis=1)
            
            # Clip minimum to prevent division by zero
            avg_test_leaf_sizes = self.xp.maximum(avg_test_leaf_sizes, 1e-5)
            
            alpha = getattr(self, "density_scaling_alpha", 1.0)
            gamma = (self.N_baseline / avg_test_leaf_sizes) ** alpha
            uq = uq * gamma
            
        # If using GPU, convert result back to a standard NumPy array for Scikit-Learn compatibility
        if self.using_gpu:
            uq = uq.get()
            
        return uq

    def _precompute_tree_paths(self, tree_idx):
        """Precomputes paths from root to all nodes in tree_idx."""
        if not hasattr(self, "tree_paths"):
            self.tree_paths = {}
            self.tree_depths = {}
            
        tree = self.estimators[tree_idx].tree_
        node_count = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        
        parent = np.full(node_count, -1, dtype=np.int32)
        depth = np.zeros(node_count, dtype=np.int32)
        
        stack = [(0, 0)]  # (node_id, current_depth)
        while stack:
            node_id, curr_depth = stack.pop()
            depth[node_id] = curr_depth
            left = children_left[node_id]
            right = children_right[node_id]
            if left != -1:
                parent[left] = node_id
                stack.append((left, curr_depth + 1))
            if right != -1:
                parent[right] = node_id
                stack.append((right, curr_depth + 1))
                
        max_depth = np.max(depth) + 1
        
        # Build path matrix
        node_paths = np.full((node_count, max_depth), -1, dtype=np.int32)
        for n in range(node_count):
            curr = n
            d = depth[n]
            while curr != -1:
                node_paths[n, d] = curr
                curr = parent[curr]
                d -= 1
                
        self.tree_paths[tree_idx] = self.xp.asarray(node_paths)
        self.tree_depths[tree_idx] = self.xp.asarray(depth)

    def compute_tree_topological_distances(self, leaf_test, leaf_train, tree_idx):
        """
        Computes the topological path distance matrix d_t(x_i, x_j) in tree tree_idx
        between a batch of test leaves and training leaves.
        """
        if not hasattr(self, "tree_paths") or tree_idx not in self.tree_paths:
            self._precompute_tree_paths(tree_idx)
            
        node_paths = self.tree_paths[tree_idx]
        node_depths = self.tree_depths[tree_idx]
        
        leaf_test_xp = self.xp.asarray(leaf_test)
        leaf_train_xp = self.xp.asarray(leaf_train)
        
        # Extract paths: (n_test, max_depth) and (n_train, max_depth)
        paths_test = node_paths[leaf_test_xp]
        paths_train = node_paths[leaf_train_xp]
        
        # Broadcast comparison: (n_test, n_train, max_depth)
        matches = paths_test[:, None, :] == paths_train[None, :, :]
        
        # Mask out padded matches (where both are -1)
        valid = (paths_test != -1)[:, None, :] & (paths_train != -1)[None, :, :]
        matches = matches & valid
        
        # LCA depth is the sum of matching nodes in prefix minus 1
        lca_depth = self.xp.sum(matches, axis=2) - 1
        
        # Extract depths
        test_depths = node_depths[leaf_test_xp][:, None]
        train_depths = node_depths[leaf_train_xp][None, :]
        
        # Distance = depth_i + depth_j - 2 * lca_depth
        d_t = test_depths + train_depths - 2 * lca_depth
        
        return d_t

    def _compute_weighted_quantile(self, values, weights, q):
        """
        Computes the weighted quantile q of values (OOB residuals) for each row in weights.
        
        Args:
            values: 1D array of shape (n_train,)
            weights: 2D array of shape (batch_len, n_train)
            q: float, quantile level (e.g. 0.025 or 0.975)
            
        Returns:
            1D array of shape (batch_len,) containing the weighted quantiles.
        """
        # Sort values (OOB residuals) and matching weights
        sort_idx = self.xp.argsort(values)
        sorted_values = values[sort_idx]
        sorted_weights = weights[:, sort_idx]
        
        # Cumulative sum of sorted weights
        cum_weights = self.xp.cumsum(sorted_weights, axis=1)
        
        # Normalize cumulative sum by the sum of weights (last column)
        sum_weights = cum_weights[:, -1:]
        
        # Avoid division by zero by clipping
        sum_weights_clipped = self.xp.maximum(sum_weights, 1e-10)
        cum_weights_norm = cum_weights / sum_weights_clipped
        
        # Find first index where cumulative probability is >= q
        idx_mask = cum_weights_norm >= q
        idx = self.xp.argmax(idx_mask, axis=1)
        
        # Detect zero-weight rows
        zero_weight_mask = (sum_weights.ravel() <= 1e-10)
        
        # Compute linear interpolation
        idx_prev = self.xp.maximum(idx - 1, 0)
        
        c_prev = cum_weights_norm[self.xp.arange(len(weights)), idx_prev]
        c_curr = cum_weights_norm[self.xp.arange(len(weights)), idx]
        
        v_prev = sorted_values[idx_prev]
        v_curr = sorted_values[idx]
        
        denom = self.xp.maximum(c_curr - c_prev, 1e-10)
        fraction = (q - c_prev) / denom
        fraction = self.xp.clip(fraction, 0.0, 1.0)
        
        val = v_prev + (v_curr - v_prev) * fraction
        
        # Fallback to standard unweighted quantile for zero-weight rows
        if self.xp.any(zero_weight_mask):
            fallback_val = self.xp.quantile(values, q)
            val = self.xp.where(zero_weight_mask, fallback_val, val)
            
        return val
