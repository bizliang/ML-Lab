import numpy as np

def train_nn(hidden_dim, num_passes=200, update_params=True, dropout_ratio=None, print_loss=None):
    """
    This function learns parameters for the neural network via backprop and batch gradient descent.

    Arguments
    ----------
    hidden_dim : Number of units in the hidden layer
    num_passes : Number of passes through the training data for gradient descent
    update_params : If True, update parameters via gradient descent
    dropout_ratio : Percentage of units to drop out
    print_loss : If integer, print the loss every integer iterations

    Returns
    -------
    params : updated model parameters (weights) stored as an unrolled vector
    grad : gradient computed from backprop stored as an unrolled vector
    """

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(1234)
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b1 = np.ones((1, hidden_dim))
    b2 = np.ones((1, output_dim))

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        '''
        Your code here
        Apply forward propagation 
        Note: apply inverted dropout on layer 2 if drapout_ratio is defined.
        # Perform inverted dropout.
        # See more info at http://cs231n.github.io/neural-networks-2/#reg
        '''
        z1 = np.dot(X, W1) + b1
        a1 = np.maximum(0, z1)  # ReLU activation
        if dropout_ratio:
            u1 = np.random.binomial(1, dropout_ratio, size=a1.shape) / dropout_ratio
            a1 *= u1
        z2 = np.dot(a1, W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Back propagation
        delta3 = probs  # Probs from forward propagation above
        delta3[range(num_examples), y] -= 1.0
        delta2 = np.dot(delta3, W2.T)
        '''
        Your code here
        Continue to Apply backward propagation (3.4)
        Note: apply inverted dropout on delta2 if drapout_ratio is defined.
        # Perform inverted dropout.
        # See more info at http://cs231n.github.io/neural-networks-2/#reg
        Calculate dW2, db2, dW1, db1
        '''
        dW2 = np.dot(a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2[z1 <= 0] = 0  # ReLU backprop
        if dropout_ratio:
            delta2 *= u1
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # Scale gradient by the number of examples and add regularization to the weight terms.
        # We do not regularize the bias terms.
        '''
        Your code here
        Follow the vanilla gradient descent procedure for performing a parameter update
        scale dW2, db2, dW1, db1

        '''
        dW2 /= num_examples
        db2 /= num_examples
        dW1 /= num_examples
        db1 /= num_examples

        if update_params:
            # Gradient descent parameter update
            lr = 0.01  # learning rate for gradient descent
            W1 += -lr * dW1
            b1 += -lr * db1
            W2 += -lr * dW2
            b2 += -lr * db2

        # Unroll model parameters and gradient and store them as a long vector
        params = np.asarray(list(chain(*[W1.flatten(),
                                         W2.flatten(),
                                         b1.flatten(),
                                         b2.flatten()
                                         ]
                                       )
                                 )
                            )
        grad = np.asarray(list(chain(*[dW1.flatten(),
                                       dW2.flatten(),
                                       db1.flatten(),
                                       db2.flatten()
                                       ]
                                     )
                               )
                          )

        # Optionally print the loss after some number of iterations
        if (print_loss is not None) and (i % print_loss == 0):
            print('Loss after iteration %i: %f' % (i, compute_loss(params)))

    return params, grad