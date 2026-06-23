# Mathematical Foundations of `minisklearn`

## 1. Gradient Descent & The Normal Equation
For Linear Regression, the Mean Squared Error (MSE) cost function is:
$$ J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\mathbf{w}^T \mathbf{x}^{(i)} + b - y^{(i)})^2 $$

### Gradient Descent Update Rule
We update parameters by moving in the direction of the negative gradient:
$$ \mathbf{w} := \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} - \alpha \frac{1}{m} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y}) $$

### Closed-Form Solution (Normal Equation)
By setting the gradient to zero, we derive the exact analytical solution:
$$ \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} $$
*Note: Implemented via `np.linalg.pinv` to handle non-invertible matrices gracefully.*

## 2. The Chain Rule for Backpropagation (MLP)
For a network with loss $L$, output activation $A^{[L]}$, and weights $W^{[l]}$, the chain rule dictates the backward pass:
$$ \frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial Z^{[l]}} \cdot \frac{\partial Z^{[l]}}{\partial W^{[l]}} = dZ^{[l]} \cdot A^{[l-1]T} $$
$$ \frac{\partial L}{\partial A^{[l-1]}} = W^{[l]T} \cdot dZ^{[l]} $$
This recursive relationship allows us to compute gradients for arbitrarily deep networks.