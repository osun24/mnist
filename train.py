import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape the data from 28x28 to 784x1
X_train = x_train.reshape(x_train.shape[0], -1).T
X_test = x_test.reshape(x_test.shape[0], -1).T

# Count number of pngs to get current trial
pngs = glob.glob("cost_and_accuracy_over_time-*.png")
trial = len(pngs) + 1

# Initialize parameters with random values
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Activation functions
# Sigmoid for hidden layers - returns a value between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Softmax used in the output layer for probabilities
def softmax(z):
    # Subtracting the max of z for numerical stability
    z_exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return z_exp / np.sum(z_exp, axis=0, keepdims=True)

# Forward propagation
def forward_pass(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)  # Use sigmoid for hidden layers
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)  # Softmax used in the output layer to display probabilities
    return Z1, A1, Z2, A2

# Cost function: Mean squared error
def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = np.sum((A2 - Y) ** 2) / m
    return cost

# Backpropagation: calculate gradients for each parameter
def backward_pass(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2

# Update parameters using gradient descent
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

# Predictions using test set 
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_pass(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=0)
    return predictions

# Plot cost and accuracy over time
def cost_and_accuracy_over_time(cost_history, accuracy_history, iteration,trial):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(cost_history)
    plt.title("Cost over time, iteration {}".format(iteration))
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.title("Accuracy over time, iteration {}".format(iteration))
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")

    # Save
    plt.savefig(f"cost_and_accuracy_over_time-{trial}.png")

# Initialize parameters
input_size = 784 # MNIST images are 28x28 pixels
hidden_size = 128 # Size of hidden layer
output_size = 10 # 10 classes (digits 0-9)
W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

# Example training loop
num_iterations = 1000
learning_rate = 0.1

# Keep track of cost and accuracy over time
cost_history = []
accuracy_history = []

# One-hot encode labels: convert from [1, 2, 3, ..., 9] to [[0, 0, 0, ..., 0], [0, 1, 0, ..., 0], ..., [0, 0, 0, ..., 1]]
def one_hot_encode(Y, num_classes):
    return np.eye(num_classes)[Y.reshape(-1)]

# Convert labels to one-hot encoding
Y_train_one_hot = one_hot_encode(y_train, output_size).T
Y_test_one_hot = one_hot_encode(y_test, output_size).T

# Training loop
for i in range(num_iterations):
    Z1, A1, Z2, A2 = forward_pass(X_train, W1, b1, W2, b2)
    cost = compute_cost(A2, Y_train_one_hot)
    cost_history.append(cost)

    dW1, db1, dW2, db2 = backward_pass(X_train, Y_train_one_hot, Z1, A1, Z2, A2, W1, W2)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    # Every 100 iterations, check accuracy on test set
    if (i + 1) % 100 == 0:
        print("Iteration {}: Cost = {}".format(i + 1, cost))
        predictions = predict(X_test, W1, b1, W2, b2)
        labels = y_test.reshape((1, -1))
        accuracy = np.mean(predictions == labels) * 100
        accuracy_history.append(accuracy)
        print("Accuracy after {} iterations: {:.2f}%".format(i + 1, accuracy))

        cost_and_accuracy_over_time(cost_history, accuracy_history, i + 1, trial)

# Save the weights
print("Saving weights...")
np.savez(f"weights-{trial}.npz", W1=W1, b1=b1, W2=W2, b2=b2)