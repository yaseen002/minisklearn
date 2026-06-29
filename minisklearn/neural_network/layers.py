import numpy as np
from minisklearn.utils.math_ops import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax

class Dense:
    """A standard fully connected neural network layer."""
    
    def __init__(self, n_in: int, n_out: int, activation: str = 'relu'):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation
        
        # HE INITIALIZATION (Crucial for ReLU to prevent vanishing/exploding gradients)
        # W ~ N(0, sqrt(2 / n_in))
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        
        # Map string to actual functions
        # Map string to actual functions
        self.activation_func = {'relu': relu, 'sigmoid': sigmoid, 'softmax': softmax, 'linear': lambda x: x}[activation]
        
        # Map string to derivatives. 
        # NOTE: The 'softmax' derivative is mathematically handled directly in mlp.py's backward pass,
        # so we just provide a dummy lambda function here to prevent a KeyError during initialization.
        self.activation_deriv = {
            'relu': relu_derivative, 
            'sigmoid': sigmoid_derivative, 
            'tanh': tanh_derivative, 
            'softmax': lambda z: np.ones_like(z), # Dummy function
            'linear': lambda x: np.ones_like(x)
        }[activation]
        
        # Cache for backpropagation
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.Z = None # Pre-activation
        self.A_prev = None # Input from previous layer

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b
        return self.activation_func(self.Z)