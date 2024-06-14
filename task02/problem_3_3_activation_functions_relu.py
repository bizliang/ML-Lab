import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    relu = np.maximum(0, z)
    drelu = np.where(z > 0, 1, 0)
    return relu, drelu


# Plot example ReLU function
fontsize = 14
z = np.linspace(-1.0, 1.0)
fz, _ = relu(z)
plt.plot(z, fz, lw=2)
plt.xlabel('$z$', fontsize=fontsize)
plt.ylabel('$f(z)$', fontsize=fontsize)
plt.title('Rectified Linear Unit (ReLU) Function', fontsize=fontsize)
plt.show()