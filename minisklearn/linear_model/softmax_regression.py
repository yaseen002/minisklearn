import numpy as np
from minisklearn.base import BaseEstimator
from minisklearn.utils.math_ops import softmax, categorical_cross_entropy

class SoftmaxRegression(BaseEstimator):
    """
    Multinomial Logistic Regression using Softmax and Gradient Descent.
    Used for multi-class classification (e.g., Iris dataset).
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 regularization: str = None, alpha: float = 0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.weights = None
        self.classes_ = None
        self.cost_history = []

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        """Converts integer class labels into one-hot encoded vectors."""
        m = y.shape[0]
        # Ensure y is 1D
        y = y.astype(int).ravel()
        n_classes = len(self.classes_)
        one_hot = np.zeros((m, n_classes))
        one_hot[np.arange(m), y] = 1
        return one_hot

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SoftmaxRegression':
        X_b = self._add_bias(X)
        m, n_features = X_b.shape
        
        # Identify unique classes (e.g., [0, 1, 2])
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Map labels to 0, 1, 2... in case they aren't already
        y_mapped = np.searchsorted(self.classes_, y)
        y_one_hot = self._one_hot(y_mapped)

        # Initialize weights matrix: shape (n_features, n_classes)
        self.weights = np.random.randn(n_features, n_classes) * 0.01

        for _ in range(self.n_iterations):
            # 1. Forward pass: Linear combination -> Softmax
            z = X_b @ self.weights
            y_pred = softmax(z) # Shape: (m, n_classes)
            
            # 2. Calculate gradients (The beautiful cancellation!)
            error = y_pred - y_one_hot
            gradients = (1/m) * (X_b.T @ error)
            
            # Add L2 regularization gradient if applicable (skip bias index 0)
            if self.regularization == 'l2':
                reg_term = (self.alpha / m) * self.weights
                reg_term[0, :] = 0 # Don't regularize bias row
                gradients += reg_term
            
            # 3. Update weights
            self.weights -= self.learning_rate * gradients
            
            # 4. Track cost
            cce = categorical_cross_entropy(y_one_hot, y_pred)
            self.cost_history.append(cce)
            
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the probability distribution across all classes."""
        X_b = self._add_bias(X)
        return softmax(X_b @ self.weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the predicted class label (the class with the highest probability)."""
        probabilities = self.predict_proba(X)
        # Get the index of the max probability, then map back to original class labels
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]