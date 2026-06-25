import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from minisklearn.linear_model.linear_regression import LinearRegression
from minisklearn.utils.data import StandardScaler

# 1. Generate simple 1D data (1 feature + 1 bias = 2 weights to plot)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Scale data (Crucial! Otherwise the loss surface is a steep trench, not a bowl)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 2. Create a meshgrid for the weights (w0 = bias, w1 = feature weight)
w0_vals = np.linspace(-2, 10, 100)
w1_vals = np.linspace(-2, 10, 100)
W0, W1 = np.meshgrid(w0_vals, w1_vals)

# 3. Calculate MSE for every combination of weights
m = len(X_scaled)
mse_grid = np.zeros_like(W0)
for i in range(len(w0_vals)):
    for j in range(len(w1_vals)):
        weights = np.array([[W0[i, j]], [W1[i, j]]])
        X_b = np.c_[np.ones((m, 1)), X_scaled]
        predictions = X_b @ weights
        mse_grid[i, j] = (1/(2*m)) * np.sum((predictions - y_scaled)**2)

# 4. Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W0, W1, mse_grid, cmap='viridis', edgecolor='none', alpha=0.8)

# 5. Plot the actual Gradient Descent path on top of the surface!
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W0, W1, mse_grid, cmap='viridis', edgecolor='none', alpha=0.8)

# --- CORRECTED LOGIC START ---
w_path = np.random.randn(2, 1) * 0.5 # Multiplied by 0.5 to start further from minimum for a better visual path
path_w0, path_w1, path_mse = [], [], [] # Initialize ALL lists as empty

X_b = np.c_[np.ones((m, 1)), X_scaled]
for _ in range(50):
    # 1. Calculate current MSE and weights for plotting BEFORE taking a step
    y_pred = X_b @ w_path
    mse = (1/(2*m)) * np.sum((y_pred - y_scaled)**2)
    
    path_w0.append(w_path[0,0])
    path_w1.append(w_path[1,0])
    path_mse.append(mse)
    
    # 2. Calculate gradients and update weights (take a step)
    gradients = (1/m) * (X_b.T @ (y_pred - y_scaled))
    w_path -= 0.1 * gradients
# --- CORRECTED LOGIC END ---

ax.plot(path_w0, path_w1, path_mse, color='red', marker='o', markersize=4, linestyle='-', label='GD Path')

ax.set_xlabel('Weight 0 (Bias)')
ax.set_ylabel('Weight 1 (Feature)')
ax.set_zlabel('Mean Squared Error (Loss)')
ax.set_title('3D Loss Surface & Gradient Descent Path')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.legend()

# Ensure the images folder exists before saving
import os
os.makedirs('docs/images', exist_ok=True)
plt.savefig('docs/images/loss_surface_3d.png', dpi=300, bbox_inches='tight')
plt.show()