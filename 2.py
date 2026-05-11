#2 Develop a Python program to implement various activation functions, including the sigmoid, tanh (hyperbolic tangent), ReLU (Rectified Linear Unit), Leaky ReLU, and softmax. The program should include functions to compute the output of each activation function for a given input. Additionally, it should be capable of plotting graphs representing the output of each activation function over a range of input values. CO1
import numpy as np
import matplotlib.pyplot as plt

# Using numpy for numerical stability and vectorization
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    # Subtracting max(x) prevents OverflowError (Numerical Stability)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Range for plotting
x = np.linspace(-10, 10, 100)

def plot_activation(name, x, y):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.title(f"{name} Activation Function")
    plt.grid(True)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.show()

# Execution
plot_activation("Sigmoid", x, sigmoid(x))
plot_activation("Tanh", x, tanh(x))
plot_activation("ReLU", x, relu(x))
plot_activation("Leaky ReLU", x, leaky_relu(x))

# Softmax visualization
z = np.array([1.0, 2.0, 3.0, 4.0, 1.0])
plt.bar(range(len(z)), softmax(z))
plt.title("Softmax Output (Probabilities)")
plt.show()
