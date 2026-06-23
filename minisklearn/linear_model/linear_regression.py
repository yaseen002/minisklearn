# minisklearn/linear_model/linear_regression.py
import numpy as np
from minisklearn.base import BaseEstimator

class LinearRegression(BaseEstimator):
    """
    Linear Regression using either the Normal Equation (Closed-Form) 
    or Gradient Descent. Supports L2 (Ridge) Regularization.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 method: str = 'gradient_descent', regularization: str = None, alpha: float = 0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.regularization = regularization # 'l2' or None
        self.alpha = alpha # Regularization strength
        self.weights = None
        self.cost_history = []

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Adds a column of ones to the front of X for the intercept term."""
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        X_b = self._add_bias(X)
        m, n = X_b.shape
        
        # Initialize weights
        self.weights = np.random.randn(n, 1) * 0.01
        y = y.reshape(-1, 1)

        if self.method == 'normal_equation':
            # Closed-form solution: w = (X^T X)^-1 X^T y
            # If L2 regularization is used, we add alpha * I to the diagonal (excluding bias)
            XtX = X_b.T @ X_b
            if self.regularization == 'l2':
                identity_matrix = np.eye(n)
                identity_matrix[0, 0] = 0 # Do not regularize the bias term
                XtX += self.alpha * identity_matrix
            
            self.weights = np.linalg.pinv(XtX) @ (X_b.T @ y)
            
        elif self.method == 'gradient_descent':
            for _ in range(self.n_iterations):
                # Forward pass: predictions
                y_pred = X_b @ self.weights
                
                # Calculate gradients
                gradients = (1/m) * (X_b.T @ (y_pred - y))
                
                # Add L2 regularization gradient if applicable (skip bias index 0)
                if self.regularization == 'l2':
                    reg_term = (self.alpha / m) * self.weights
                    reg_term[0] = 0 # Don't regularize bias
                    gradients += reg_term
                
                # Update weights
                self.weights -= self.learning_rate * gradients
                
                # Track cost for plotting
                mse = (1/(2*m)) * np.sum((y_pred - y)**2)
                self.cost_history.append(mse)
        else:
            raise ValueError("Method must be 'normal_equation' or 'gradient_descent'")
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = self._add_bias(X)
        return X_b @ self.weights