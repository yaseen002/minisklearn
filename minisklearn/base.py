from abc import ABC, abstractmethod
import numpy as np

class BaseEstimator(ABC):
    """Abstract base class for all estimators in minisklearn."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEstimator':
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the given data."""
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 or accuracy."""
        # Default implementation can be overridden by subclasses
        predictions = self.predict(X)
        if y.ndim == 1 or y.shape[1] == 1: # Regression or Binary
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        else: # Multi-class classification
            return np.mean(predictions == np.argmax(y, axis=1))