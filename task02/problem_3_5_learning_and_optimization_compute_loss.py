import numpy as np
import matplotlib.pyplot as plt


def compute_loss(params, X, y, input_dim, hidden_dim, output_dim, weight_decay=0.0005):
    """ Function to compute the average loss over the dataset. """

    W1, W2, b1, b2 = reshape(params, input_dim, hidden_dim, output_dim)

    # Forward propagation
    probs = forward(params, X, input_dim, hidden_dim, output_dim, predict=False)

    # Number of samples
    N = X.shape[0]

    # Compute the data loss using cross-entropy
    corect_logprobs = -np.log(probs[range(N), y])
    data_loss = np.sum(corect_logprobs) / N

    # Compute the regularization loss
    reg_loss = 0.5 * weight_decay * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    # Total loss including regularization
    loss = data_loss + reg_loss

    return loss

def softmax(z):
    """Helper function to compute the element-wise softmax activation of an array."""
    # Shift argument to prevent potential numerical instability from large exponentials
    exp = np.exp(z - np.max(z))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    return probs

def relu(z):
    return np.maximum(0, z)

def forward(params, x, input_dim, hidden_dim, output_dim, predict=False):
    """The forward pass to predict softmax probability distribution over class labels."""

    W1, W2, b1, b2 = reshape(params, input_dim, hidden_dim, output_dim)

    # Forward propagation
    z2 = np.dot(x, W1) + b1
    # applying the activation function with relu
    a2 = relu(z2)
    z3 = np.dot(a2, W2) + b2
    probs = softmax(z3)

    return np.argmax(probs, axis=1) if predict else probs

# Example usage
input_dim = 2  # example input dimension
hidden_dim = 500  # example hidden layer dimension
output_dim = 3  # example output dimension

# Randomly initialize parameters for testing
params = np.random.randn(input_dim * hidden_dim + hidden_dim * output_dim + hidden_dim + output_dim)
x = np.random.randn(1, input_dim)  # example input

# Forward pass
probs = forward(params, x, input_dim, hidden_dim, output_dim)
print(probs)

def reshape(array, input_dim, hidden_dim, output_dim):
    W1 = np.reshape(array[0:input_dim * hidden_dim],
                    (input_dim, hidden_dim))
    W2 = np.reshape(array[input_dim * hidden_dim:hidden_dim * (input_dim + output_dim)],
                    (hidden_dim, output_dim))
    b1 = array[hidden_dim * (input_dim + output_dim):hidden_dim * (input_dim + output_dim + 1)]
    b2 = array[hidden_dim * (input_dim + output_dim + 1):]

    return W1, W2, b1, b2
