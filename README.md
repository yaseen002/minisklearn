# minisklearn: A From-Scratch Machine Learning Library

> **Core Philosophy:** *Derive before you code.* 
> `minisklearn` is a lightweight, educational machine learning library implemented entirely from scratch using only **Python** and **NumPy**. There are no high-level frameworks (PyTorch, TensorFlow) or automated ML libraries (scikit-learn) used in the core algorithms.

## 🎯 Project Objective
The primary goal of this project is to bridge the gap between theoretical mathematics and software engineering. By manually implementing the linear algebra, calculus (chain rule), and optimization techniques (Gradient Descent) that drive modern AI, this library serves as a rigorous proof of mathematical maturity and algorithmic understanding.

Every line of code in the core algorithms is a direct translation of mathematical proofs documented in the [`docs/math/`](docs/math/) directory.

## 🛠️ Key Engineering & Mathematical Features
- **Strict API Design:** Implements an abstract `BaseEstimator` pattern mimicking the scikit-learn API for consistency.
- **Numerical Stability:** Core math utilities are engineered to prevent floating-point errors (e.g., Log-Sum-Exp trick for Softmax, probability clipping for Cross-Entropy, and split-domain calculation for Sigmoid).
- **Geometric Intuition:** Includes tooling to visualize optimization landscapes (e.g., 3D Loss Surface plotting for Gradient Descent).
- **Type Safety & Testing:** Fully type-hinted (PEP-484) with strict input shape validation and gradient checking.
