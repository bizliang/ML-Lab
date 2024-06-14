import numpy as np
import matplotlib.pyplot as plt


def compute_numerical_gradient(func, params):
    eps = 1e-4
    num_grad = np.zeros(len(params))
    E = np.eye(len(params))
    for i in range(len(params)):
        params_plus = params + eps * E[:, i]
        params_minus = params - eps * E[:, i]
        num_grad[i] = (func(params_plus) - func(params_minus)) / (2.0 * eps)

    return num_grad


def quadratic_function(x):
    return x[0]**2 + 3 * x[0] * x[1]


def quadratic_function_prime(x):
    grad = np.zeros(2)
    grad[0] = 2 * x[0] + 3 * x[1]
    grad[1] = 3 * x[0]
    return grad


def plot_gradients(analytical_grad, numerical_grad, label1, label2):
    error_thresh = np.mean(np.abs(analytical_grad - numerical_grad))

    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.5
    error_config = {'ecolor': '0.3'}

    index = np.arange(len(analytical_grad))

    # Plot bar charts
    rects1 = ax.bar(index, analytical_grad, bar_width,
                    alpha=opacity, color='green',
                    error_kw=error_config, label=label1)

    rects2 = ax.bar(index + bar_width, numerical_grad, bar_width,
                    alpha=opacity, color='blue',
                    error_kw=error_config, label=label2)

    # Plot horizontal line indicating error threshold
    ax.plot(index, np.repeat(error_thresh, repeats=len(analytical_grad)),
            "r-", linewidth=2, label='Mean Error')

    fontsize = 14
    ax.set_ylabel('Gradient Value', fontsize=fontsize)
    ax.set_title(' '.join([label1, 'vs.', label2, 'Gradient']), fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False  # labels along the bottom edge are off
    )
    fig.tight_layout()
    # save the plot image
    plt.savefig('problem_3_8_numerical_gradient_checking_plt_fig.png')
    plt.show()


# Package imports
import seaborn as sns
sns.set()
sns.set_style('darkgrid')
from sklearn.datasets import make_classification
from itertools import chain

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
                           weights=None, flip_y=0.01, class_sep=0.85, hypercube=True, shift=0.0,
                           scale=1.0, shuffle=True, random_state=0)


def softmax(z):
    # Shift argument to prevent potential numerical instability from large exponentials
    exp = np.exp(z - np.max(z))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs


def relu(z):
    return np.maximum(0, z)


def reshape(array, input_dim, hidden_dim, output_dim):
    W1 = np.reshape(array[0:input_dim * hidden_dim],
                    (input_dim, hidden_dim))
    W2 = np.reshape(array[input_dim * hidden_dim:hidden_dim * (input_dim + output_dim)],
                    (hidden_dim, output_dim))
    b1 = array[hidden_dim * (input_dim + output_dim):hidden_dim * (input_dim + output_dim + 1)]
    b2 = array[hidden_dim * (input_dim + output_dim + 1):]

    return W1, W2, b1, b2


def forward(params, x, input_dim, hidden_dim, output_dim, predict=False):
    W1, W2, b1, b2 = reshape(params, input_dim, hidden_dim, output_dim)

    # Forward propagation
    z2 = np.dot(x, W1) + b1
    # applying the activation function with relu
    a2 = relu(z2)
    z3 = np.dot(a2, W2) + b2
    probs = softmax(z3)

    return np.argmax(probs, axis=1) if predict else probs


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


def train_nn(hidden_dim, num_passes=200, update_params=True, dropout_ratio=None, print_loss=None):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(1234)
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b1 = np.ones((1, hidden_dim))
    b2 = np.ones((1, output_dim))

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
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
        dW2 = np.dot(a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, W2.T)
        delta2[z1 <= 0] = 0  # ReLU backprop
        if dropout_ratio:
            delta2 *= u1
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # Scale gradient by the number of examples and add regularization to the weight terms.
        dW2 /= num_examples
        db2 /= num_examples
        dW1 /= num_examples
        db1 /= num_examples

        # We do not regularize the bias terms.
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


# perform numerical gradient checking of the neural network implementation
num_examples = X.shape[0] # number of examples
input_dim = X.shape[1] # input layer dimensionality
hidden_dim = 3 # hidden layer dimensionality
output_dim = len(np.unique(y)) # output layer dimensionality

weight_decay = 0.0005 # regularization strength, smaller values provide greater strength

params, grad = train_nn(hidden_dim, update_params=False, dropout_ratio=None, num_passes=1)
num_grad = compute_numerical_gradient(lambda p: compute_loss(p, X, y, input_dim, hidden_dim, output_dim, weight_decay), params)
abs_error = np.abs(grad - num_grad)

plot_gradients(grad, num_grad, 'Backprop', 'Numerical')

print('\nMean Absolute Error: {:.12e}'.format(np.mean(abs_error)))

