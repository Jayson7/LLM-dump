#   Generate a Simple Dataset

import numpy as np
import matplotlib.pyplot as plt

# iniate a randome seed as usual 
np.random.seed(42)

# messy codes done mind my bad habit
# Generate class 0
X0 = np.random.normal(2, 1, (50, 2))  # centered at (2,2)
y0 = np.zeros((50, 1))                # label 0

# Generate class 1
X1 = np.random.normal(4, 1, (50, 2))  # centered at (4,4)
y1 = np.ones((50, 1))                 # label 1

# Combine all my data
X = np.vstack((X0, X1))               # shape (100, 2)
y = np.vstack((y0, y1))               # shape (100, 1)

# Plot the data
plt.scatter(X[:50, 0], X[:50, 1], label="Class 0")
plt.scatter(X[50:, 0], X[50:, 1], label="Class 1")
plt.legend()
plt.title("Synthetic Binary Classification Data")
plt.show()

#   Sigmoid Function

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#   Initialize Parameters

m, n = X.shape
X_b = np.c_[np.ones((m, 1)), X]  # Add bias term (column of 1s)
theta = np.zeros((n + 1, 1))     # Initialize weights (including bias)

#   Train Using Gradient Descent

lr = 0.1
epochs = 1000

for epoch in range(epochs):
    z = X_b.dot(theta)                 # Linear combination
    h = sigmoid(z)                     # Prediction using sigmoid

    error = h - y                      # Error between prediction and actual
    gradient = X_b.T.dot(error) / m    # Gradient of loss
    theta -= lr * gradient             # Update parameters

#   Prediction Function

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    proba = sigmoid(X_b.dot(theta))
    return (proba >= 0.5).astype(int)  # Convert probabilities to class 0 or 1

#   Evaluate Accuracy

y_pred = predict(X, theta)
accuracy = np.mean(y_pred == y)
print(f"Model accuracy: {accuracy * 100:.2f}%")

