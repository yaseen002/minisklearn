import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as sk_linear
import sklearn.cluster as sk_cluster
import sklearn.neural_network as sk_nn

# Import our library
from minisklearn.linear_model.linear_regression import LinearRegression
from minisklearn.linear_model.logistic_regression import LogisticRegression
from minisklearn.cluster.kmeans import KMeans
from minisklearn.neural_network.mlp import MLP
from minisklearn.metrics.supervised import r2_score, accuracy_score

print("Loading and preprocessing datasets...")

# 1. Regression: California Housing (Subset for speed)
california = fetch_california_housing()
X_reg, y_reg = california.data[:2000], california.target[:2000]
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler_reg = StandardScaler().fit(X_reg_train)
X_reg_train_s, X_reg_test_s = scaler_reg.transform(X_reg_train), scaler_reg.transform(X_reg_test)

# 2. Binary Classification: Breast Cancer
cancer = load_breast_cancer()
X_bin, y_bin = cancer.data, cancer.target
X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)
scaler_bin = StandardScaler().fit(X_bin_train)
X_bin_train_s, X_bin_test_s = scaler_bin.transform(X_bin_train), scaler_bin.transform(X_bin_test)

# 3. Clustering: Iris (Unsupervised, so we drop labels)
iris = load_iris()
X_clus = StandardScaler().fit_transform(iris.data)

# 4. Multi-class/Deep Learning: Digits (8x8 images)
digits = load_digits()
X_dig, y_dig = digits.data / 16.0, digits.target # Scale pixels to 0-1
X_dig_train, X_dig_test, y_dig_train, y_dig_test = train_test_split(X_dig, y_dig, test_size=0.2, random_state=42)

results = {'Models': [], 'Ours': [], 'Sklearn': [], 'Time (s)': []}

print("Running Benchmarks...")

# --- Benchmark 1: Linear Regression ---
# FIX: Use Normal Equation (exact solution) instead of Gradient Descent
start = time.time()
our_lr = LinearRegression(method='normal_equation').fit(X_reg_train_s, y_reg_train)
our_r2 = r2_score(y_reg_test, our_lr.predict(X_reg_test_s))
time_ours = time.time() - start

start = time.time()
sk_lr = sk_linear.LinearRegression().fit(X_reg_train_s, y_reg_train)
sk_r2 = r2_score(y_reg_test, sk_lr.predict(X_reg_test_s))
time_sk = time.time() - start

results['Models'].append('Linear Reg (R²)')
results['Ours'].append(our_r2)
results['Sklearn'].append(sk_r2)
results['Time (s)'].append(f"{time_ours:.3f} / {time_sk:.3f}")

# --- Benchmark 2: Logistic Regression ---
# FIX: Lower learning rate (0.01) and more iterations (2000)
start = time.time()
our_log = LogisticRegression(learning_rate=0.1, n_iterations=3000).fit(X_bin_train_s, y_bin_train)
our_acc = accuracy_score(y_bin_test, our_log.predict(X_bin_test_s))
time_ours = time.time() - start

start = time.time()
sk_log = sk_linear.LogisticRegression(max_iter=2000).fit(X_bin_train_s, y_bin_train)
sk_acc = accuracy_score(y_bin_test, sk_log.predict(X_bin_test_s))
time_sk = time.time() - start

results['Models'].append('Logistic Reg (Acc)')
results['Ours'].append(our_acc)
results['Sklearn'].append(sk_acc)
results['Time (s)'].append(f"{time_ours:.3f} / {time_sk:.3f}")

# --- Benchmark 3: K-Means ---
start = time.time()
our_km = KMeans(n_clusters=3, random_state=42, max_iter=300).fit(X_clus)
our_inertia = our_km.inertia_
time_ours = time.time() - start

start = time.time()
sk_km = sk_cluster.KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_clus)
sk_inertia = sk_km.inertia_
time_sk = time.time() - start

results['Models'].append('K-Means (Inertia)')
results['Ours'].append(our_inertia)
results['Sklearn'].append(sk_inertia)
results['Time (s)'].append(f"{time_ours:.3f} / {time_sk:.3f}")

# --- Benchmark 4: MLP (Digits) ---
# FIX: Lower learning rate (0.01) and more epochs (1000)
y_dig_train_oh = np.zeros((len(y_dig_train), 10))
y_dig_train_oh[np.arange(len(y_dig_train)), y_dig_train] = 1

start = time.time()
our_mlp = MLP(layer_dims=[64, 64, 10], learning_rate=0.05, epochs=2000).fit(X_dig_train, y_dig_train)
our_dig_acc = accuracy_score(y_dig_test, our_mlp.predict(X_dig_test))
time_ours = time.time() - start

start = time.time()
sk_mlp = sk_nn.MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42, learning_rate_init=0.01).fit(X_dig_train, y_dig_train)
sk_dig_acc = accuracy_score(y_dig_test, sk_mlp.predict(X_dig_test))
time_sk = time.time() - start

results['Models'].append('MLP Digits (Acc)')
results['Ours'].append(our_dig_acc)
results['Sklearn'].append(sk_dig_acc)
results['Time (s)'].append(f"{time_ours:.3f} / {time_sk:.3f}")

# --- Plotting the Results ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('minisklearn vs scikit-learn: Performance Benchmark', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    model_name = results['Models'][i]
    our_val = results['Ours'][i]
    sk_val = results['Sklearn'][i]
    
    bars = ax.bar(['minisklearn', 'scikit-learn'], [our_val, sk_val], color=['#1f77b4', '#ff7f0e'], width=0.6)
    ax.set_title(model_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)
        
    # Add time annotation
    time_str = results['Time (s)'][i]
    ax.text(0.5, -0.15, f'Train Time (Ours / Sklearn): {time_str}s', transform=ax.transAxes, ha='center', fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs('docs/images', exist_ok=True)
plt.savefig('docs/images/benchmark_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- Final Results Summary ---")
for i in range(4):
    print(f"{results['Models'][i]} | Ours: {results['Ours'][i]:.4f} | Sklearn: {results['Sklearn'][i]:.4f}")