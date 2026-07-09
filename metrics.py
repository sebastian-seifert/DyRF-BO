import numpy as np
from scipy.spatial.distance import jensenshannon

def calculate_jensen_shannon_divergence(uncertainty, y_true_binary, n_bins=50):
    """FIXED: Now squares the distance value to return true JSD in [0, 1]"""
    u_id = uncertainty[y_true_binary == 0]
    u_ood = uncertainty[y_true_binary == 1]

    if len(u_id) < 2 or len(u_ood) < 2: return np.nan

    u_min = np.min(uncertainty)
    u_max = np.max(uncertainty)
    if u_max - u_min < 1e-10:
        return 0.0

    # Determine the number of bins.
    # If using Freedman-Diaconis ("fd" or "auto"):
    # PROS:
    # - Dynamic scaling: Adapts to sample size N via N^(1/3), resolving sparseness in small datasets.
    # - Outlier robustness: Uses Interquartile Range (IQR), focusing bins on bulk probability mass.
    # CONS:
    # - Run-to-run comparison bias: Dynamic bin numbers change the maximum possible entropy of 
    #   the histogram, rendering raw JSD comparisons between different evaluations slightly biased.
    # - Multimodal under-binning: Tends to over-smooth in distributions with sharp isolated peaks.
    if isinstance(n_bins, str) and n_bins.lower() in ["fd", "auto"]:
        q75, q25 = np.percentile(uncertainty, [75, 25])
        iqr = q75 - q25
        n = len(uncertainty)
        if iqr > 1e-10 and n > 0:
            bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
            n_bins_resolved = int(np.ceil((u_max - u_min) / bin_width))
            # Clip between 2 and 200 to prevent empty-bin variance or memory exhaustion
            n_bins_resolved = int(np.clip(n_bins_resolved, 2, 200))
        else:
            n_bins_resolved = 50
    else:
        n_bins_resolved = int(n_bins)

    bin_edges = np.linspace(u_min, u_max, n_bins_resolved + 1)

    p_id, _ = np.histogram(u_id, bins=bin_edges)
    p_ood, _ = np.histogram(u_ood, bins=bin_edges)

    # 1. Convert counts to probability distribution and add epsilon to avoid log(0)
    p_id = (p_id / np.sum(p_id)) + 1e-10
    p_ood = (p_ood / np.sum(p_ood)) + 1e-10

    # 2. Re-normalize to ensure they sum to exactly 1.0 (strict requirement of scipy.jensenshannon)
    p_id /= np.sum(p_id)
    p_ood /= np.sum(p_ood)

    js_distance = jensenshannon(p_id, p_ood, base=2.0)
    return float(js_distance ** 2)

def calculate_mutual_information(uncertainty, y_true_binary, n_bins=50):
    """
    Computes Normalized Mutual Information (NMI) using discrete binning.
    Guarantees output is in [0, 1] and eliminates resubstitution bias.
    """
    n_total = len(uncertainty)
    if n_total < 3 or np.min(y_true_binary) == np.max(y_true_binary):
        return np.nan

    # 1. Discretize the continuous uncertainty into bins
    u_min, u_max = np.min(uncertainty), np.max(uncertainty)
    if u_max - u_min < 1e-10:
        return 0.0 # Constant uncertainty carries 0 information
        
    # Resolve bin count using Freedman-Diaconis rule if requested
    if isinstance(n_bins, str) and n_bins.lower() in ["fd", "auto"]:
        q75, q25 = np.percentile(uncertainty, [75, 25])
        iqr = q75 - q25
        if iqr > 1e-10 and n_total > 0:
            bin_width = 2.0 * iqr / (n_total ** (1.0 / 3.0))
            n_bins_resolved = int(np.ceil((u_max - u_min) / bin_width))
            n_bins_resolved = int(np.clip(n_bins_resolved, 2, 200))
        else:
            n_bins_resolved = 50
    else:
        n_bins_resolved = int(n_bins)

    bin_edges = np.linspace(u_min, u_max, n_bins_resolved + 1)
    # Map each uncertainty value to its bin index (1 to n_bins_resolved)
    u_discrete = np.digitize(uncertainty, bin_edges) - 1
    # Clip boundaries
    u_discrete = np.clip(u_discrete, 0, n_bins_resolved - 1)

    # 2. Compute joint and marginal distributions
    joint_counts, _, _ = np.histogram2d(u_discrete, y_true_binary, 
                                        bins=[n_bins_resolved, 2], 
                                        range=[[0, n_bins_resolved], [0, 2]])
    
    P_joint = joint_counts / n_total
    P_u = np.sum(P_joint, axis=1)
    P_y = np.sum(P_joint, axis=0)

    # 3. Calculate Shannon Entropies in bits
    # H(Y)
    P_y_nonzero = P_y[P_y > 0]
    h_y = -np.sum(P_y_nonzero * np.log2(P_y_nonzero))
    if h_y < 1e-10:
        return np.nan

    # H(U)
    P_u_nonzero = P_u[P_u > 0]
    h_u = -np.sum(P_u_nonzero * np.log2(P_u_nonzero))

    # H(U, Y)
    P_joint_nonzero = P_joint[P_joint > 0]
    h_uy = -np.sum(P_joint_nonzero * np.log2(P_joint_nonzero))

    # 4. MI = H(U) + H(Y) - H(U, Y)
    mi = h_u + h_y - h_uy
    
    # 5. Return Uncertainty Coefficient (fraction of H(Y) explained by U) bounded in [0, 1]
    return float(np.clip(mi / h_y, 0.0, 1.0))

def calculate_rejection_curve(uncertainty, predictions, y_true, rejection_rates, loss_type="MSE"):
    """
    Computes the regression loss (MSE or MAE) at each rejection rate p.
    Points are rejected in descending order of predicted uncertainty.
    """
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)
    uncertainty = np.asarray(uncertainty)
    
    n_samples = len(uncertainty)
    sorted_indices = np.argsort(uncertainty)[::-1] # Descending order of uncertainty
    
    if loss_type == "MSE":
        errors = (predictions - y_true) ** 2
    elif loss_type == "MAE":
        errors = np.abs(predictions - y_true)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    losses = []
    for p in rejection_rates:
        n_reject = int(np.floor(p * n_samples))
        if n_reject >= n_samples:
            n_reject = n_samples - 1
        keep_indices = sorted_indices[n_reject:]
        losses.append(np.mean(errors[keep_indices]))
        
    return np.array(losses)

def calculate_aurc(rejection_rates, losses):
    """
    Computes the Area Under the Rejection Curve (AURC) using trapezoidal integration.
    """
    rejection_rates = np.asarray(rejection_rates)
    losses = np.asarray(losses)
    dx = np.diff(rejection_rates)
    mean_y = 0.5 * (losses[:-1] + losses[1:])
    return float(np.sum(mean_y * dx))

def calculate_oracle_rejection_curve(predictions, y_true, rejection_rates, loss_type="MSE"):
    """
    Computes the Oracle (perfect UQ) rejection curve by sorting by actual error magnitude.
    """
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)
    
    if loss_type == "MSE":
        errors = (predictions - y_true) ** 2
    elif loss_type == "MAE":
        errors = np.abs(predictions - y_true)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    return calculate_rejection_curve(
        uncertainty=errors,
        predictions=predictions,
        y_true=y_true,
        rejection_rates=rejection_rates,
        loss_type=loss_type
    )

def calculate_random_rejection_curve(predictions, y_true, rejection_rates, loss_type="MSE", n_shuffles=20, random_state=42):
    """
    Computes the Random rejection curve baseline analytically as a constant line of the overall error.
    This is mathematically exact and eliminates the noise of shuffling.
    """
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)
    rejection_rates = np.asarray(rejection_rates)
    
    if loss_type == "MSE":
        overall_error = np.mean((predictions - y_true) ** 2)
    elif loss_type == "MAE":
        overall_error = np.mean(np.abs(predictions - y_true))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    return np.full(len(rejection_rates), overall_error)

def calculate_aurc_exact(uncertainty, predictions, y_true, p_max=0.95, loss_type="MSE"):
    """
    Computes the exact Area Under the Rejection Curve (AURC) analytically
    from p=0 to p=p_max, without grid discretization or numerical integration error.
    Complexity is O(N log N) due to sorting.
    """
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)
    uncertainty = np.asarray(uncertainty)
    n_samples = len(uncertainty)
    
    if n_samples == 0:
        return 0.0
        
    # Sort the errors in descending order of predicted uncertainty
    abs_uncertainty = np.abs(uncertainty)
    sorted_indices = np.argsort(abs_uncertainty)[::-1]
    
    if loss_type == "MSE":
        errors = (predictions - y_true) ** 2
    elif loss_type == "MAE":
        errors = np.abs(predictions - y_true)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    sorted_errors = errors[sorted_indices]
    
    # Compute the remaining means using cumulative sums from right to left
    cumsum_right = np.cumsum(sorted_errors[::-1])[::-1]
    counts = np.arange(n_samples, 0, -1)
    losses = cumsum_right / counts
    
    # Rejection curve is constant on [k/N, (k+1)/N) with value losses[k]
    K_max = int(np.floor(p_max * n_samples))
    if K_max >= n_samples:
        K_max = n_samples - 1

    #Numerical Intergation step 
    aurc = np.sum(losses[:K_max]) / n_samples + losses[K_max] * (p_max - K_max / n_samples)
    return float(aurc)

def calculate_naurc(rejection_rates, rejection_curve, oracle_curve, random_curve):
    """
    Computes the Normalized Area Under the Rejection Curve (NAURC) bounded in [0, 5].
    NAURC = (AURC_model - AURC_oracle) / (AURC_random - AURC_oracle)
    0.0 represents perfect UQ (matching oracle), 1.0 represents random baseline.
    """
    aurc_model = calculate_aurc(rejection_rates, rejection_curve)
    aurc_oracle = calculate_aurc(rejection_rates, oracle_curve)
    aurc_random = calculate_aurc(rejection_rates, random_curve)
    
    denom = aurc_random - aurc_oracle
    if denom < 1e-10:
        return 0.0
    return float(np.clip((aurc_model - aurc_oracle) / denom, 0.0, 5.0))

def calculate_roc_metrics(y_true_binary, uncertainty):
    """
    Computes both AUROC and FPR@95TPR in a single pass using a single call to roc_curve.
    Returns:
        auroc (float), fpr95 (float)
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true_binary, uncertainty)
    auroc = float(auc(fpr, tpr))
    idx = np.argmax(tpr >= 0.95)
    fpr95 = float(fpr[idx])
    return auroc, fpr95


def calculate_aupr(y_true_binary, uncertainty):
    """
    Computes the Area Under the Precision-Recall Curve (AUPR) using average_precision_score.
    """
    from sklearn.metrics import average_precision_score
    y_true_binary = np.asarray(y_true_binary)
    uncertainty = np.asarray(uncertainty)
    if len(np.unique(y_true_binary)) < 2:
        return np.nan
    return float(average_precision_score(y_true_binary, uncertainty))
