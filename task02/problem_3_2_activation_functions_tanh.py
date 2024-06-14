import numpy as np
import matplotlib.pyplot as plt


def tanh(z):
    tanh = np.tanh(z)
    dtanh = 1 - np.square(tanh)
    return tanh, dtanh


# Plot example tanh function
fontsize = 14
z = np.linspace(-5, 5)
fz, _ = tanh(z)
plt.plot(z, fz, lw=2)
plt.xlabel('$z$', fontsize=fontsize)
plt.ylabel('$f(z)$', fontsize=fontsize)
plt.title('Hyperbolic Tangent (Tanh) Function', fontsize=fontsize)
plt.show()