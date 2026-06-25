import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from minisklearn.linear_model.softmax_regression import SoftmaxRegression
from minisklearn.utils.data import StandardScaler
import os
import matplotlib as mpl

# 1. Load Iris dataset and extract ONLY 2 features for 2D plotting
iris = load_iris()
# Feature 2: Petal Length, Feature 3: Petal Width
X = iris.data[:, 2:] 
y = iris.target
class_names = iris.target_names

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train the Softmax Regression model
model = SoftmaxRegression(learning_rate=0.1, n_iterations=1000, regularization='l2', alpha=0.01)
model.fit(X_scaled, y)

# Print final accuracy
accuracy = np.mean(model.predict(X_scaled) == y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# 3. Create a mesh grid for the 2D plot
h = 0.05
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict the class for every point in the mesh
X_mesh = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(X_mesh)
Z = Z.reshape(xx.shape)

# 4. Plot the 3-class decision boundary
fig, ax = plt.subplots(figsize=(10, 7))
# Contour plot fills the background with the 3 predicted classes
# We use a custom colormap for the 3 classes
# Modern Matplotlib syntax for discrete colormaps:
cmap_light = mpl.colormaps['Pastel1'].resampled(3)
cmap_bold = mpl.colormaps['Dark2'].resampled(3)

ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k', cmap=cmap_bold, s=60)

ax.set_title('Softmax Regression: Iris Dataset Decision Boundaries\n(Petal Length vs. Petal Width)', fontsize=14)
ax.set_xlabel('Petal Length (Scaled)')
ax.set_ylabel('Petal Width (Scaled)')

# Add a legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cmap_bold(i), edgecolor='k', label=class_names[i]) for i in range(3)]
ax.legend(handles=legend_elements, loc='upper left')

os.makedirs('docs/images', exist_ok=True)
plt.savefig('docs/images/decision_boundary_softmax_iris.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. BONUS: Plot the Loss Curve to prove convergence
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(model.cost_history, color='purple', linewidth=2)
ax2.set_title('Categorical Cross-Entropy Loss Convergence', fontsize=14)
ax2.set_xlabel('Iterations (Epochs)')
ax2.set_ylabel('Loss')
ax2.grid(True, linestyle='--', alpha=0.7)
plt.savefig('docs/images/softmax_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()