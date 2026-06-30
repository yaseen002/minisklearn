import numpy as np
from minisklearn.base import BaseEstimator
from minisklearn.utils.math_ops import sigmoid, binary_cross_entropy

class LogisticRegression(BaseEstimator):
    """
    Binary Logistic Regression using Gradient Descent.
    Supports L2 (Ridge) Regularization.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 regularization: str = None, alpha: float = 0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization # 'l2' or None
        self.alpha = alpha
        self.weights = None
        self.cost_history = []

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Adds a column of ones to the front of X for the intercept term."""
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        X_b = self._add_bias(X)
        m, n = X_b.shape
        
        # Initialize weights
        self.weights = np.random.randn(n, 1) * 0.01
        y = y.reshape(-1, 1)

        for _ in range(self.n_iterations):
            # Forward pass: linear combination + sigmoid
            z = X_b @ self.weights
            y_pred = sigmoid(z)
            
            # Calculate gradients (Notice how similar this is to Linear Regression!)
            error = y_pred - y
            gradients = (1/m) * (X_b.T @ error)
            
            # Add L2 regularization gradient if applicable (skip bias index 0)
            if self.regularization == 'l2':
                reg_term = (self.alpha / m) * self.weights
                reg_term[0] = 0 # Don't regularize bias
                gradients += reg_term
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            # Track cost for plotting
            bce = binary_cross_entropy(y, y_pred)
            self.cost_history.append(bce)
            
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the probability of the positive class."""
        X_b = self._add_bias(X)
        return sigmoid(X_b @ self.weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int).ravel() # <--- Added .ravel()