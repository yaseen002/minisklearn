import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the mean Silhouette Coefficient of all samples.
    Implemented from scratch using only NumPy.
    """
    unique_labels = np.unique(labels)
    n_samples = X.shape[0]
    silhouette_vals = np.zeros(n_samples)
    
    # Precompute pairwise distance matrix for the whole dataset
    # Using the expansion trick: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    dist_matrix = np.sqrt(np.clip(X_sq + X_sq.T - 2 * (X @ X.T), 0, None))
    
    for i in range(n_samples):
        label_i = labels[i]
        
        # Calculate 'a': mean distance to other points in the SAME cluster
        same_cluster_mask = (labels == label_i)
        same_cluster_mask[i] = False # Exclude the point itself
        same_cluster_dists = dist_matrix[i, same_cluster_mask]
        
        if np.sum(same_cluster_mask) == 0:
            silhouette_vals[i] = 0
            continue
            
        a_i = np.mean(same_cluster_dists)
        
        # Calculate 'b': mean distance to points in the NEAREST DIFFERENT cluster
        b_i = np.inf
        for k in unique_labels:
            if k == label_i:
                continue
            other_cluster_mask = (labels == k)
            other_cluster_dists = dist_matrix[i, other_cluster_mask]
            mean_dist_k = np.mean(other_cluster_dists)
            if mean_dist_k < b_i:
                b_i = mean_dist_k
                
        # Calculate silhouette for this point
        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        
    return np.mean(silhouette_vals)