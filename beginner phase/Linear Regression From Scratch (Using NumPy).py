# Linear Regression From Scratch (Using NumPy)

import numpy as np 
import matplotlib.pyplot as plt

# using the fomular y=mx+b

#random seed to ensure the same random numbers are generated each time


# Random seed for reproducibility
np.random.seed(0)

# Create synthetic data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # true m=3, b=4

plt.scatter(X, y)
plt.title("Generated Data")
plt.show()


# Implement Gradient Descent for Linear Regression

# Add bias (intercept) column
X_b = np.c_[np.ones((100, 1)), X]  # shape (100, 2)

# Initialize parameters
theta = np.random.randn(2, 1)  # [b, m]

# Learning rate and iterations
lr = 0.1
epochs = 1000

for epoch in range(epochs):
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    theta -= lr * gradients

print("Estimated theta (b, m):", theta)



plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red')
plt.title("Linear Regression Fit")
plt.show()
