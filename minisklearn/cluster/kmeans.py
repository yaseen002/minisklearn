import numpy as np
from minisklearn.base import BaseEstimator

class KMeans(BaseEstimator):
    """
    K-Means Clustering with K-Means++ initialization.
    Minimizes the Within-Cluster Sum of Squares (Inertia).
    """
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4, random_state: int = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol # Convergence tolerance
        self.random_state = random_state
        
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None # Within-Cluster Sum of Squares
        self.n_iter_ = 0

    def _euclidean_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Calculates pairwise Euclidean distance between rows of X1 and X2."""
        # Using the expansion: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        # This is much faster and more memory-efficient than nested loops.
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        cross_term = X1 @ X2.T
        distances_sq = X1_sq + X2_sq.T - 2 * cross_term
        # Clip to prevent negative numbers due to floating point inaccuracies
        return np.sqrt(np.clip(distances_sq, 0, None))

    def _kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """
        K-Means++ initialization.
        Selects initial centroids to be spread out, avoiding poor local minima.
        """
        rng = np.random.RandomState(self.random_state)
        m, n = X.shape
        centroids = np.empty((self.n_clusters, n))
        
        # 1. Choose the first centroid uniformly at random
        centroids[0] = X[rng.randint(0, m)]
        
        for k in range(1, self.n_clusters):
            # 2. Compute distances from each point to the nearest existing centroid
            dists = self._euclidean_distance(X, centroids[:k])
            min_dists = np.min(dists, axis=1)
            
            # 3. Square the distances (this is the core of K-Means++)
            probs = min_dists ** 2
            probs /= np.sum(probs) # Normalize to create a probability distribution
            
            # 4. Choose the next centroid based on these probabilities
            next_centroid_idx = rng.choice(m, p=probs)
            centroids[k] = X[next_centroid_idx]
            
        return centroids

    def fit(self, X: np.ndarray, y=None) -> 'KMeans':
        # Note: y is ignored, included for API compatibility with BaseEstimator
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        m, n = X.shape
        
        # Initialize centroids using the smart K-Means++ method
        self.centroids = self._kmeans_plus_plus(X)
        
        for i in range(self.max_iter):
            # --- E-Step: Assign points to nearest centroid ---
            distances = self._euclidean_distance(X, self.centroids)
            self.labels_ = np.argmin(distances, axis=1)
            
            # --- M-Step: Update centroids to the mean of assigned points ---
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                cluster_points = X[self.labels_ == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    # Edge case: Empty cluster. Re-initialize it to a random point.
                    new_centroids[k] = X[np.random.randint(0, m)]
            
            # --- Check for convergence ---
            centroid_shift = np.sum(self._euclidean_distance(new_centroids, self.centroids))
            self.centroids = new_centroids
            self.n_iter_ = i + 1
            
            if centroid_shift < self.tol:
                break
                
        # Calculate final Inertia (WCSS)
        self.inertia_ = self._compute_inertia(X)
        return self

    def _compute_inertia(self, X: np.ndarray) -> float:
        """Computes the Within-Cluster Sum of Squares (WCSS)."""
        distances = self._euclidean_distance(X, self.centroids)
        min_dists = np.min(distances, axis=1)
        return np.sum(min_dists ** 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigns new data points to the nearest existing centroid."""
        distances = self._euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)