import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    out = np.zeros_like(z, dtype=float)
    pos_mask = z >= 0
    out[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    neg_mask = ~pos_mask
    exp_z = np.exp(z[neg_mask])
    out[neg_mask] = exp_z / (1 + exp_z)
    return out

def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))"""
    return a * (1 - a)

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax using the log-sum-exp trick."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit."""
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if z > 0, else 0."""
    return (z > 0).astype(float)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(a: np.ndarray) -> np.ndarray:
    return 1 - a**2

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Numerically stable Binary Cross-Entropy."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Numerically stable Categorical Cross-Entropy for multi-class."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))