# tests/test_phase1.py
import sys
import os
import numpy as np

# --- THE FIX: Add the project root to the system path ---
# This gets the absolute path of the 'tests' directory, goes up one level ('..'), 
# and adds it to Python's search path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# --------------------------------------------------------

from minisklearn.utils.math_ops import sigmoid, softmax, binary_cross_entropy
from minisklearn.utils.data import StandardScaler, train_test_split

# 1. Test Numerical Stability of Sigmoid
print("--- Testing Sigmoid Stability ---")
large_neg = np.array([-1000, -500, 0, 500, 1000])
print(f"Input: {large_neg}")
print(f"Output: {sigmoid(large_neg)}") 

# 2. Test Softmax
print("\n--- Testing Softmax ---")
z = np.array([[1000, 1000, 1000], [1, 2, 3]]) 
print(f"Softmax Output:\n{softmax(z)}") 

# 3. Test Scaler and Split
print("\n--- Testing Data Utils ---")
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 0, 1, 1, 1])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled Mean (should be ~0): {np.mean(X_scaled, axis=0)}")
print(f"Scaled Std (should be ~1): {np.std(X_scaled, axis=0)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")