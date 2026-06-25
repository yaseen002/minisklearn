import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from minisklearn.cluster.kmeans import KMeans
from minisklearn.metrics.unsupervised import silhouette_score
from minisklearn.utils.data import StandardScaler
import os

# 1. Generate synthetic data with 4 distinct blobs
X, y_true = make_blobs(n_samples=800, centers=4, cluster_std=0.60, random_state=42)

# Scale data (Crucial for distance-based algorithms!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Run K-Means for K = 1 to 10 to find the "Elbow"
K_range = range(1, 11)
inertias = []
silhouettes = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    
    # Silhouette score is undefined for K=1, so we skip it
    if k > 1:
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    else:
        silhouettes.append(0)

# 3. Plot the Elbow Curve and Silhouette Scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, marker='o', linestyle='--', color='b', linewidth=2)
ax1.set_title('The Elbow Method (Inertia)', fontsize=14)
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Within-Cluster Sum of Squares (Inertia)')
ax1.set_xticks(K_range)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(K_range[1:], silhouettes[1:], marker='s', linestyle='-', color='g', linewidth=2)
ax2.set_title('Silhouette Score Analysis', fontsize=14)
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Mean Silhouette Coefficient')
ax2.set_xticks(K_range)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
os.makedirs('docs/images', exist_ok=True)
plt.savefig('docs/images/kmeans_elbow_silhouette.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Fit the final model with the optimal K (which we know is 4)
optimal_k = 4
final_model = KMeans(n_clusters=optimal_k, random_state=42)
final_model.fit(X_scaled)

# 5. Visualize the final clusters and centroids
fig2, ax3 = plt.subplots(figsize=(9, 7))

# Scatter the data points, colored by their predicted cluster
scatter = ax3.scatter(X_scaled[:, 0], X_scaled[:, 1], c=final_model.labels_, 
                      cmap='viridis', edgecolor='k', s=40, alpha=0.7)

# Plot the centroids as large red 'X's
ax3.scatter(final_model.centroids[:, 0], final_model.centroids[:, 1], 
            marker='x', s=200, linewidths=3, color='red', label='Centroids')

ax3.set_title(f'K-Means Clustering (K={optimal_k})\nFrom-Scratch Implementation', fontsize=14)
ax3.set_xlabel('Feature 1 (Scaled)')
ax3.set_ylabel('Feature 2 (Scaled)')
ax3.legend()

plt.savefig('docs/images/kmeans_final_clusters.png', dpi=300, bbox_inches='tight')
plt.show()