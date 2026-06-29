import numpy as np
from minisklearn.base import BaseEstimator
from minisklearn.neural_network.layers import Dense
from minisklearn.utils.math_ops import softmax, categorical_cross_entropy

class MLP(BaseEstimator):
    """Multi-Layer Perceptron with manual backpropagation."""
    
    def __init__(self, layer_dims: list, learning_rate: float = 0.01, epochs: int = 1000):
        """
        layer_dims: e.g., [2, 16, 16, 3] means 2 inputs, two hidden layers of 16, 3 output classes.
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []
        self.cost_history = []
        
        # Build the network architecture
        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i+1]
            # Use ReLU for hidden layers, Softmax for the final output layer
            act = 'relu' if i < len(layer_dims) - 2 else 'softmax'
            self.layers.append(Dense(n_in, n_out, activation=act))

    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A # This is the final probability distribution (Y_hat)

    def _backward_pass(self, Y_true_one_hot: np.ndarray) -> None:
        m = Y_true_one_hot.shape[0]
        
        # 1. Calculate the gradient of the Loss w.r.t the final output (dA_L)
        # Because we use Softmax + Categorical Cross-Entropy, the derivative simplifies beautifully!
        Y_hat = self.layers[-1].activation_func(self.layers[-1].Z)
        dA = Y_hat - Y_true_one_hot 
        
        # 2. Backpropagate through the layers in reverse
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            # If it's the output layer, dA is already calculated above.
            # If it's a hidden layer, we need to calculate dA from the next layer's dZ.
            if i < len(self.layers) - 1:
                next_layer = self.layers[i+1]
                # Chain rule: dA_current = dZ_next @ W_next.T
                dA = dZ_next @ next_layer.W.T
                
            # Calculate dZ for current layer
            # For Softmax, the derivative is handled in the dA step above, so we just pass dA as dZ
            if layer.activation_name == 'softmax':
                dZ = dA 
            else:
                dZ = dA * layer.activation_deriv(layer.Z)
                
            # Calculate gradients for Weights and Biases
            layer.dW = (1/m) * (layer.A_prev.T @ dZ)
            layer.db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            # Save dZ for the next iteration (which is the previous layer in the network)
            dZ_next = dZ

    def _update_weights(self) -> None:
        for layer in self.layers:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP':
        # One-hot encode targets
        classes = np.unique(y)
        m = X.shape[0]
        Y_one_hot = np.zeros((m, len(classes)))
        Y_one_hot[np.arange(m), y.astype(int)] = 1

        for epoch in range(self.epochs):
            # Forward
            Y_hat = self._forward_pass(X)
            
            # Calculate Loss
            loss = categorical_cross_entropy(Y_one_hot, Y_hat)
            self.cost_history.append(loss)
            
            # Backward
            self._backward_pass(Y_one_hot)
            
            # Update
            self._update_weights()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")
                
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward_pass(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)