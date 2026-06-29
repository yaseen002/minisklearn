import numpy as np
from minisklearn.neural_network.mlp import MLP
from minisklearn.utils.math_ops import categorical_cross_entropy

def gradient_check(model: MLP, X: np.ndarray, y_one_hot: np.ndarray, epsilon: float = 1e-7) -> bool:
    """
    Verifies the analytical gradients from backprop against numerical gradients.
    """
    print("Running Gradient Check...")
    
    # 1. Get analytical gradients via backprop
    model._forward_pass(X)
    model._backward_pass(y_one_hot)
    
    max_relative_error = 0.0
    
    # 2. Check every weight and bias in every layer
    for l_idx, layer in enumerate(model.layers):
        for param_name in ['W', 'b']:
            param = getattr(layer, param_name)
            analytical_grad = getattr(layer, f'd{param_name}')
            
            # Iterate through every single number in the matrix
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                
                # Save original value
                original_val = param[idx]
                
                # Calculate J(theta + epsilon)
                param[idx] = original_val + epsilon
                y_hat_plus = model._forward_pass(X)
                loss_plus = categorical_cross_entropy(y_one_hot, y_hat_plus)
                
                # Calculate J(theta - epsilon)
                param[idx] = original_val - epsilon
                y_hat_minus = model._forward_pass(X)
                loss_minus = categorical_cross_entropy(y_one_hot, y_hat_minus)
                
                # Numerical gradient
                numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                
                # Restore original value
                param[idx] = original_val
                
                # Calculate relative error
                numerator = np.abs(numerical_grad - analytical_grad[idx])
                denominator = np.abs(numerical_grad) + np.abs(analytical_grad[idx])
                relative_error = numerator / denominator if denominator > 0 else 0.0
                
                if relative_error > max_relative_error:
                    max_relative_error = relative_error
                    
                it.iternext()
                
    print(f"Max Relative Error: {max_relative_error:.2e}")
    if max_relative_error < 1e-5: # 1e-5 is a safe threshold for float64
        print("✅ GRADIENT CHECK PASSED! Backpropagation is mathematically flawless.")
        return True
    else:
        print("❌ GRADIENT CHECK FAILED! There is a bug in the chain rule.")
        return False