# 1 Write a python program to implement the learning rules such as Hebbian Learning Rule, Perceptron Learning Rule, Delta Learning Rule, Correlation Learning Rule, and OutStar Learning Rule in deep learning. CO1

import numpy as np
np.random.seed(42)
N = 3 # input size
#Weight Initialisation
weights = {
    "hebbian"    : np.random.rand(N, N),
    "perceptron" : np.random.rand(N),
    "delta"      : np.random.rand(N, N),
    "correlation": np.random.rand(N, N),
    "outstar"    : np.random.rand(N, N),
}
# hebbian : Wij = Xi-Xj
def hebbian(W, x):
    return W + np.outer(x,x)

# perceptron: Δw = p.(t-y^).x
def perceptron(W, x, t, lr=0.1):
    pred = 1 if np.dot(W, x) >= 0 else -1  #step activation
    error = t - pred
    return W + lr * error* x

#delta: Δw = p.(t-y^).x^T (linear output, matrix)
def delta(W, x, t, lr=0.1):
    error = t - W @ x          # vector error
    return W + lr * np.outer(error, x)

# Correlation: Δwij = p.Xi.dj  (desired output d = x here)
def correlation(W, x, d, lr=0.1):
    return W + lr * np.outer(x, d)    # d is desired output

# Outstar: Wjk += p.(yk − Wjk) if active, else 0
def outstar(W, x, active=True, lr=0.1):
    return W + lr * np.outer(x - W @ x, x) if active else W

# ----- Sample Data -----
x = np.array([1.0, 0.5, -1.0])          # input pattern
t_scalar = 1                           # perceptron target (+1)
t_vector = np.array([0.8, 0.2, 0.6])  # delta / outstar target
d = np.array([1.0, 0.0, 1.0])        # correlation desired output

# ----- Apply Learning Rules -----
weights["hebbian"] = hebbian(weights["hebbian"], x)
weights["perceptron"] = perceptron(weights["perceptron"], x, t_scalar)
weights["delta"] = delta(weights["delta"], x, t_vector)
weights["correlation"] = correlation(weights["correlation"], x, d)
weights["outstar"] = outstar(weights["outstar"], x, active=True)

# ----- Print Results -----
for name, W in weights.items():
    print(f"\n{name.capitalize()} Weights ({W.shape}):\n{np.round(W, 4)}")
