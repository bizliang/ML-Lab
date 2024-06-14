import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('darkgrid')
from sklearn.datasets import make_classification
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

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

def softmax(z):
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
    z2 = np.dot(x, W1) + b1
    a2 = relu(z2)
    z3 = np.dot(a2, W2) + b2
    probs = softmax(z3)
    return np.argmax(probs, axis=1) if predict else probs

def compute_loss(params, X, y, input_dim, hidden_dim, output_dim, weight_decay=0.0005):
    W1, W2, b1, b2 = reshape(params, input_dim, hidden_dim, output_dim)
    probs = forward(params, X, input_dim, hidden_dim, output_dim, predict=False)
    N = X.shape[0]
    corect_logprobs = -np.log(probs[range(N), y])
    data_loss = np.sum(corect_logprobs) / N
    reg_loss = 0.5 * weight_decay * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    loss = data_loss + reg_loss
    return loss

def train_nn(X_train, y_train, hidden_dim, num_passes=200, update_params=True, dropout_ratio=None, print_loss=None):
    np.random.seed(1234)
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    num_examples = X_train.shape[0]
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b1 = np.ones((1, hidden_dim))
    b2 = np.ones((1, output_dim))
    for i in range(0, num_passes):
        z1 = np.dot(X_train, W1) + b1
        a1 = np.maximum(0, z1)
        if dropout_ratio:
            u1 = np.random.binomial(1, dropout_ratio, size=a1.shape) / dropout_ratio
            a1 *= u1
        z2 = np.dot(a1, W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        delta3 = probs
        delta3[range(num_examples), y_train] -= 1.0
        dW2 = np.dot(a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, W2.T)
        delta2[z1 <= 0] = 0
        if dropout_ratio:
            delta2 *= u1
        dW1 = np.dot(X_train.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)
        dW2 /= num_examples
        db2 /= num_examples
        dW1 /= num_examples
        db1 /= num_examples
        if update_params:
            lr = 0.01
            W1 += -lr * dW1
            b1 += -lr * db1
            W2 += -lr * dW2
            b2 += -lr * db2
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
        if (print_loss is not None) and (i % print_loss == 0):
            print('Loss after iteration %i: %f' % (i, compute_loss(params, X_train, y_train, input_dim, hidden_dim, output_dim)))
    return params, grad

def plot_decision_boundary(pred_func):
    x_min, x_max = X_val[:, 0].min() - 0.5, X_val[:, 0].max() + 0.5
    y_min, y_max = X_val[:, 1].min() - 0.5, X_val[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colormap, alpha=0.4)
    plt.scatter(X_val[:, 0], X_val[:, 1], s=60, c=y_val, cmap=colormap)

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
                           weights=None, flip_y=0.01, class_sep=0.85, hypercube=True, shift=0.0,
                           scale=1.0, shuffle=True, random_state=0)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.33, random_state=1234)
palette = sns.color_palette('deep', 3)
colormap = ListedColormap(palette)

# Overfit the neural network by using 200 units in the hidden layer
hidden_dim = 200

# Train the neural network
params, _ = train_nn(X, y, hidden_dim, num_passes=20000, dropout_ratio=None, print_loss=1000)
nn_preds = forward(params, X_val, X.shape[1], hidden_dim, len(np.unique(y)), predict=True)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plot_decision_boundary(lambda x: forward(params, x, X.shape[1], hidden_dim, len(np.unique(y)), predict=True))
plt.title('Hidden Layer Size %d - Validation Accuracy: %.6f' % (hidden_dim, np.mean(y_val == nn_preds)), fontsize=15)
# save the plot image
plt.savefig('problem_4_4_overfitting_plt_fig.png')
plt.show()

# Print classification report
print('\nClassification Report for Neural Network with Hidden Layer Size 200:\n')
print(classification_report(y_val, nn_preds))
print('Validation Accuracy for Neural Network with Hidden Layer Size 200: {:.6f}'.format(np.mean(y_val == nn_preds)))
