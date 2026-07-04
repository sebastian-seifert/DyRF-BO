import numpy as np

def generate_data(func_dict, func_name, seed, points_per_dim=None, gap_type='empty', sparse_multiplier=12, scaling_law='linear', min_samples_leaf=5):
    """
    Generates training and test data. Uses grid-based meshes for 1D-5D and
    fallback random uniform sampling for >=6D to avoid exponential complexity.
    """
    rng = np.random.default_rng(seed)
    func = func_dict[func_name]
    func_obj = func["func"]
    gap = func["gap"]
    x_range = func["range"]

    # Determine dimensionality dynamically from lambda argument count
    ndim = func_obj.__code__.co_argcount

    # If ndim >= 3, use random uniform sampling instead of dense grids to prevent OOM
    if ndim >= 3:
        # Scale number of samples with dimension to maintain a reasonable dataset size
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
        else: # 10D
            n_samples = 10000
            
        X = rng.uniform(x_range[0], x_range[1], size=(n_samples, ndim))
    else:
        # Dynamic default points_per_dim depending on dimension to control exponential explosion
        if points_per_dim is None:
            if ndim == 1:
                points_per_dim = 1200
            elif ndim == 2:
                points_per_dim = 50
            else:
                points_per_dim = 30

        # Generate the coordinate grids for each axis
        grids = [np.linspace(x_range[0], x_range[1], points_per_dim) for _ in range(ndim)]
        meshes = np.meshgrid(*grids, indexing='ij')
        X = np.stack([m.ravel() for m in meshes], axis=1)
        n_samples = len(X)
    
    # Create the multidimensional OOD gap mask (Hypercube) for training set generation
    gap_mask_train = np.ones(len(X), dtype=bool)
    for d in range(ndim):
        gap_mask_train &= (X[:, d] >= gap[0]) & (X[:, d] <= gap[1])

    if gap_type == 'sparse':
        # Apply the chosen OOD gap sparsity scaling law
        if scaling_law == 'linear':
            n_keep = sparse_multiplier * ndim
        # to account for the high volume in high dimensions
        elif scaling_law == 'fractional':
            scaling_factor = 1.3
            n_keep = int(sparse_multiplier * (scaling_factor ** ndim))
        # number of required samples based on the minimum samples per leaf in a decision tree
        elif scaling_law == 'leaf':
            n_keep = int(sparse_multiplier * min_samples_leaf)
        else:
            n_keep = sparse_multiplier * ndim
        
        # Train mask excludes all points inside the gap
        train_mask = ~gap_mask_train
        
        # Generate n_keep points uniformly sampled across the entire hypercube gap
        gap_rng = np.random.default_rng(seed + 100000)
        X_sparse_gap = gap_rng.uniform(gap[0], gap[1], size=(n_keep, ndim))
        
        X_train = np.concatenate([X[train_mask], X_sparse_gap], axis=0)
    else:
        # gap_type == 'empty'
        train_mask = ~gap_mask_train
        X_train = X[train_mask]
    
    # calculate y_train labels based on the current function
    y_train = func_obj(*[X_train[:, d] for d in range(ndim)]).ravel()
    noise = 0.1
    y_train += rng.normal(0, noise, len(y_train))

    # Construct test set by explicitly sampling ID and OOD points (preserving hypercube structure)
    # Note: Test size is approximatly equal to training size, which could be a big bottleneck!!
    id_split = 0.7
    n_id = int(n_samples * id_split)
    n_ood = n_samples - n_id
    
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
    y_test = func_obj(*[X_test[:, d] for d in range(ndim)]).ravel()
    y_test += rng.normal(0, noise, len(y_test))
    
    y_true_binary = np.zeros(len(X_test), dtype=int)
    y_true_binary[n_id:] = 1

    return X_train, y_train, X_test, y_test, y_true_binary
