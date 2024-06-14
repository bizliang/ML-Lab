import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    """Helper function to compute the element-wise softmax activation of an array."""
    # Shift argument to prevent potential numerical instability from large exponentials
    exp = np.exp(z - np.max(z))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    return probs


def reshape(array, input_dim, hidden_dim, output_dim):
    W1 = np.reshape(array[0:input_dim * hidden_dim],
                    (input_dim, hidden_dim))
    W2 = np.reshape(array[input_dim * hidden_dim:hidden_dim * (input_dim + output_dim)],
                    (hidden_dim, output_dim))
    b1 = array[hidden_dim * (input_dim + output_dim):hidden_dim * (input_dim + output_dim + 1)]
    b2 = array[hidden_dim * (input_dim + output_dim + 1):]

    return W1, W2, b1, b2


def forward(params, x, predict=False):
    """The forward pass to predict softmax probability distribution over class labels."""

    W1, W2, b1, b2 = reshape(params)

    # Forward propagation
    '''
    Your code here
    follow 3.1 expression and input your code
    Note: 
    W1, W2 are the weights
    b1, b2 are the bias
    probs is your 3rd layer value
    '''

    return np.argmax(probs, axis=1) if predict else probs
