import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    sigmoid = 1 / (1 + np.exp(-z))
    dsigmoid = sigmoid * (1 - sigmoid)
    return sigmoid, dsigmoid


# Plot example sigmoid function
fontsize = 14
z = np.linspace(-10, 10)
fz, _ = sigmoid(z)
plt.plot(z, fz, lw=2)
plt.xlabel('$z$', fontsize=fontsize)
plt.ylabel('$f(z)$', fontsize=fontsize)
plt.title('Sigmoid Function', fontsize=fontsize)
plt.show()