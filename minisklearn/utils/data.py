import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None) -> tuple:
    """Splits data into train and test sets."""
    if random_state is not None:
        np.random.seed(random_state)
        
    indices = np.random.permutation(X.shape[0])
    test_set_size = int(X.shape[0] * test_size)
    
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Prevent division by zero for constant features
        self.std_[self.std_ == 0] = 1.0 
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)