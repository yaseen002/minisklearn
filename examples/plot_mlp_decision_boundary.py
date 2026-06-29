import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from minisklearn.neural_network.mlp import MLP
from minisklearn.neural_network.gradient_check import gradient_check
from minisklearn.utils.data import StandardScaler
import os

# 1. Generate non-linear "Moons" dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode for gradient check
Y_one_hot = np.zeros((len(y), 2))
Y_one_hot[np.arange(len(y)), y] = 1

# 2. Run Gradient Check on a small subset to prove our math!
print("--- Running Gradient Check ---")
gradient_check(MLP([2, 4, 2], learning_rate=0.01, epochs=10), X_scaled[:10], Y_one_hot[:10])
print("\n--- Training MLP ---")

# 3. Train the MLP 
# Architecture: 2 inputs -> 16 hidden -> 16 hidden -> 2 outputs
model = MLP(layer_dims=[2, 16, 16, 2], learning_rate=0.05, epochs=3000)
model.fit(X_scaled, y)

# 4. Plot the Non-Linear Decision Boundary
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

X_mesh = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(X_mesh)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 7))
ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu, s=30)

ax.set_title('Multi-Layer Perceptron: Non-Linear Decision Boundary (Moons Dataset)', fontsize=14)
ax.set_xlabel('Feature 1 (Scaled)')
ax.set_ylabel('Feature 2 (Scaled)')

os.makedirs('docs/images', exist_ok=True)
plt.savefig('docs/images/mlp_nonlinear_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Plot Loss Curve
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(model.cost_history, color='darkorange', linewidth=2)
ax2.set_title('MLP Training Loss (Categorical Cross-Entropy)', fontsize=14)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.grid(True, linestyle='--', alpha=0.7)
plt.savefig('docs/images/mlp_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()