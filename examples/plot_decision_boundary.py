import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from minisklearn.linear_model.logistic_regression import LogisticRegression
from minisklearn.utils.data import StandardScaler
import os

# 1. Generate a clean 2D dataset with 2 classes
X, y = make_blobs(n_samples=500, centers=2, random_state=42, cluster_std=1.5)
y = y.reshape(-1, 1)

# Scale data (Crucial for Logistic Regression convergence)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train the Logistic Regression model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000, regularization='l2', alpha=0.1)
model.fit(X_scaled, y)

# 3. Create a mesh grid to predict every single point in the 2D space
h = 0.02 # Step size in the mesh
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict the class for every point in the mesh
X_mesh = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(X_mesh)
Z = Z.reshape(xx.shape)

# 4. Plot the contour (the decision boundary) and the data points
fig, ax = plt.subplots(figsize=(10, 7))
# The contourf fills the background with the predicted classes
ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
# Scatter the actual data points on top
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y.ravel(), edgecolors='k', cmap=plt.cm.RdYlBu)

ax.set_title('Logistic Regression: Linear Decision Boundary', fontsize=14)
ax.set_xlabel('Feature 1 (Scaled)')
ax.set_ylabel('Feature 2 (Scaled)')

# Save the image for the README
os.makedirs('docs/images', exist_ok=True)
plt.savefig('docs/images/decision_boundary_logistic.png', dpi=300, bbox_inches='tight')
plt.show()