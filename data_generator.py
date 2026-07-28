import numpy as np

def generate_manifold_data(func_obj, x_range, gap, ndim, n_samples, gap_type, sparse_multiplier, scaling_law, min_samples_leaf, seed):
    rng = np.random.default_rng(seed)
    noise = 0.1
    
    # 0D manifold in 1D space
    if ndim == 1:
        c = (x_range[0] + x_range[1]) / 2.0
        L = 0.25 * (x_range[1] - x_range[0])
        
        # ID training data: close to the point c
        X_train = rng.normal(c, 0.05, size=(n_samples, 1))
        
        # Test set: ID and OOD
        id_split = 0.7
        n_id = int(n_samples * id_split)
        n_ood = n_samples - n_id
        
        X_id = rng.normal(c, 0.05, size=(n_id, 1))
        
        # OOD: translated off the point by lambda
        # Sample lambda from [-1.5 * L, -0.8 * L] U [0.8 * L, 1.5 * L]
        half_ood = n_ood // 2
        lambda_pos = rng.uniform(0.8 * L, 1.5 * L, size=half_ood)
        lambda_neg = rng.uniform(-1.5 * L, -0.8 * L, size=n_ood - half_ood)
        lambdas = np.concatenate([lambda_pos, lambda_neg])
        rng.shuffle(lambdas)
        
        X_ood = c + lambdas[:, np.newaxis] + rng.normal(0, 0.05, size=(n_ood, 1))
        X_test = np.concatenate([X_id, X_ood], axis=0)
        n_id_actual = n_id
        
    else:
        # Codimension-1 manifold: dimension d = ndim - 1
        d = ndim - 1
        
        # Define nonlinear manifold function f(z)
        center = (x_range[0] + x_range[1]) / 2.0
        width = x_range[1] - x_range[0]
        A = 0.2 * width
        omega = 2.0 * np.pi / width
        
        def f(z):
            # z is shape (N, d)
            return center + (A / d) * np.sum(np.sin(omega * z), axis=1)
            
        def grad_f(z):
            # df/dz_i is shape (N, d)
            return (A * omega / d) * np.cos(omega * z)
            
        # Sample latent coordinates z uniformly
        # ID training coordinates:
        # We need to handle gaps in z_1
        z_min, z_max = x_range[0], x_range[1]
        
        # Oversample to ensure we get enough points after filtering
        z_raw = rng.uniform(z_min, z_max, size=(n_samples * 10, d))
        in_gap = (z_raw[:, 0] >= gap[0]) & (z_raw[:, 0] <= gap[1])
        
        if gap_type == 'sparse':
            if scaling_law == 'linear':
                n_keep = sparse_multiplier * ndim
            elif scaling_law == 'fractional':
                scaling_factor = 1.3
                n_keep = int(sparse_multiplier * (scaling_factor ** ndim))
            elif scaling_law == 'leaf':
                n_keep = int(sparse_multiplier * min_samples_leaf)
            else:
                n_keep = sparse_multiplier * ndim
                
            z_id = z_raw[~in_gap][:n_samples - n_keep]
            # Sample sparse gap coordinates
            z_gap = rng.uniform(gap[0], gap[1], size=(n_keep, d))
            # The remaining coordinates are sampled uniformly in the gap
            for col in range(1, d):
                z_gap[:, col] = rng.uniform(z_min, z_max, size=n_keep)
                
            z_train = np.concatenate([z_id, z_gap], axis=0)
        else:
            # gap_type == 'empty'
            z_train = z_raw[~in_gap][:n_samples]
            
        # Map training coordinates to D-dimensional space: x = (z_1, ..., z_d, f(z))
        f_train = f(z_train)
        X_train_manifold = np.column_stack([z_train, f_train])
        X_train = X_train_manifold + rng.normal(0, 0.05, size=X_train_manifold.shape)
        
        # Now construct test set: ID (labeled 0) and OOD (labeled 1)
        id_split = 0.7
        n_id = int(n_samples * id_split)
        n_ood = n_samples - n_id
        
        # ID test coordinates (sampled without gap constraint)
        z_test_id = rng.uniform(z_min, z_max, size=(n_id, d))
        f_test_id = f(z_test_id)
        X_test_id_manifold = np.column_stack([z_test_id, f_test_id])
        X_id = X_test_id_manifold + rng.normal(0, 0.05, size=X_test_id_manifold.shape)
        n_id_actual = len(X_id)
        
        # OOD test coordinates: translate along normal vectors
        z_test_ood = rng.uniform(z_min, z_max, size=(n_ood, d))
        f_test_ood = f(z_test_ood)
        X_test_ood_manifold = np.column_stack([z_test_ood, f_test_ood])
        
        # Normal vectors calculation:
        # g(x) = x_D - f(z) = 0
        # grad g = (-df/dz_1, ..., -df/dz_d, 1)
        df_dz = grad_f(z_test_ood) # (n_ood, d)
        grad_g = np.column_stack([-df_dz, np.ones(n_ood)]) # (n_ood, D)
        norm_grad_g = np.linalg.norm(grad_g, axis=1, keepdims=True) # (n_ood, 1)
        v = grad_g / norm_grad_g # unit normals
        
        # Translation distance lambda
        # L = 0.25 * width
        L = 0.25 * width
        half_ood = n_ood // 2
        lambda_pos = rng.uniform(0.8 * L, 1.5 * L, size=half_ood)
        lambda_neg = rng.uniform(-1.5 * L, -0.8 * L, size=n_ood - half_ood)
        lambdas = np.concatenate([lambda_pos, lambda_neg])
        rng.shuffle(lambdas)
        
        X_ood = X_test_ood_manifold + lambdas[:, np.newaxis] * v + rng.normal(0, 0.05, size=X_test_ood_manifold.shape)
        X_test = np.concatenate([X_id, X_ood], axis=0)
        
    # Evaluate clean labels first
    y_train_raw = func_obj(*[X_train[:, d] for d in range(ndim)]).ravel()
    y_test_raw = func_obj(*[X_test[:, d] for d in range(ndim)]).ravel()
    
    # Normalize target function outputs to unit variance based on train statistics
    std_y = np.std(y_train_raw)
    if std_y > 1e-8:
        mean_y = np.mean(y_train_raw)
        y_train_clean = (y_train_raw - mean_y) / std_y
        y_test_clean = (y_test_raw - mean_y) / std_y
    else:
        y_train_clean = y_train_raw
        y_test_clean = y_test_raw
        
    y_train = y_train_clean + rng.normal(0, noise, len(y_train_clean))
    y_test = y_test_clean + rng.normal(0, noise, len(y_test_clean))
    
    y_true_binary = np.zeros(len(X_test), dtype=int)
    y_true_binary[n_id_actual:] = 1
    
    return X_train, y_train, X_test, y_test, y_true_binary

def generate_data(func_dict, func_name, seed, points_per_dim=None, gap_type='empty', sparse_multiplier=12, scaling_law='linear', min_samples_leaf=5, ood_type='hypercube'):
    """
    Generates training and test data. Uses grid-based meshes for 1D-2D and
    fallback random uniform sampling for >=3D to avoid exponential complexity.
    Supports either hypercube boundary OOD or low-dimensional manifold OOD testing.
    """
    rng = np.random.default_rng(seed)
    func = func_dict[func_name]
    func_obj = func["func"]
    gap = func["gap"]
    x_range = func["range"]

    # Determine dimensionality dynamically from lambda argument count
    ndim = func_obj.__code__.co_argcount

    # Determine number of samples
    if ndim >= 3:
        if ndim == 3:
            n_samples = 3000
        elif ndim == 4:
            n_samples = 4000
        elif ndim == 5:
            n_samples = 5000
        elif ndim == 6:
            n_samples = 6000
        elif ndim == 7:
            n_samples = 7000
        elif ndim == 8:
            n_samples = 8000
        elif ndim == 9:
            n_samples = 9000
        else:
            n_samples = ndim * 1000
    else:
        if points_per_dim is None:
            if ndim == 1:
                points_per_dim = 1200
            elif ndim == 2:
                points_per_dim = 50
            else:
                points_per_dim = 30
        n_samples = (points_per_dim ** ndim)

    if ood_type == 'manifold':
        return generate_manifold_data(
            func_obj=func_obj,
            x_range=x_range,
            gap=gap,
            ndim=ndim,
            n_samples=n_samples,
            gap_type=gap_type,
            sparse_multiplier=sparse_multiplier,
            scaling_law=scaling_law,
            min_samples_leaf=min_samples_leaf,
            seed=seed
        )

    # Fallback to original hypercube behavior
    if ndim >= 3:
        X = rng.uniform(x_range[0], x_range[1], size=(n_samples, ndim))
    else:
        grids = [np.linspace(x_range[0], x_range[1], points_per_dim) for _ in range(ndim)]
        meshes = np.meshgrid(*grids, indexing='ij')
        X = np.stack([m.ravel() for m in meshes], axis=1)
    
    gap_mask_train = np.ones(len(X), dtype=bool)
    for d in range(ndim):
        gap_mask_train &= (X[:, d] >= gap[0]) & (X[:, d] <= gap[1])

    if gap_type == 'sparse':
        if scaling_law == 'linear':
            n_keep = sparse_multiplier * ndim
        elif scaling_law == 'fractional':
            scaling_factor = 1.3
            n_keep = int(sparse_multiplier * (scaling_factor ** ndim))
        elif scaling_law == 'leaf':
            n_keep = int(sparse_multiplier * min_samples_leaf)
        else:
            n_keep = sparse_multiplier * ndim
        
        train_mask = ~gap_mask_train
        gap_rng = np.random.default_rng(seed + 100000)
        X_sparse_gap = gap_rng.uniform(gap[0], gap[1], size=(n_keep, ndim))
        X_train = np.concatenate([X[train_mask], X_sparse_gap], axis=0)
    else:
        train_mask = ~gap_mask_train
        X_train = X[train_mask]
    
    # Construct test set by explicitly sampling ID and OOD points (capped at min(n_samples * 0.3, 1000))
    id_split = 0.7
    n_test = min(int(n_samples * 0.3), 1000)
    n_id = int(n_test * id_split)
    n_ood = n_test - n_id
    
    X_ood = rng.uniform(gap[0], gap[1], size=(n_ood, ndim))
    
    X_id = []
    while len(X_id) < n_id:
        batch = rng.uniform(x_range[0], x_range[1], size=(n_id - len(X_id), ndim))
        batch_in_gap = np.ones(len(batch), dtype=bool)
        for d in range(ndim):
            batch_in_gap &= (batch[:, d] >= gap[0]) & (batch[:, d] <= gap[1])
        X_id.extend(batch[~batch_in_gap])
    X_id = np.array(X_id)
    
    X_test = np.concatenate([X_id, X_ood], axis=0)
    
    y_train_raw = func_obj(*[X_train[:, d] for d in range(ndim)]).ravel()
    y_test_raw = func_obj(*[X_test[:, d] for d in range(ndim)]).ravel()
    
    # Normalize target function outputs to unit variance based on train statistics
    std_y = np.std(y_train_raw)
    if std_y > 1e-8:
        mean_y = np.mean(y_train_raw)
        y_train_clean = (y_train_raw - mean_y) / std_y
        y_test_clean = (y_test_raw - mean_y) / std_y
    else:
        y_train_clean = y_train_raw
        y_test_clean = y_test_raw
        
    noise = 0.1
    y_train = y_train_clean + rng.normal(0, noise, len(y_train_clean))
    
    y_true_binary = np.zeros(len(X_test), dtype=int)
    y_true_binary[n_id:] = 1

    y_test = y_test_clean + rng.normal(0, noise, len(y_test_clean))
    return X_train, y_train, X_test, y_test, y_true_binary
