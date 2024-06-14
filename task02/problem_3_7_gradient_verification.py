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
    plt.savefig('problem_3_7_gradient_verification_plt_fig.png')
    plt.show()


x = np.array([4.0, 10.0]) # 2D vector input

grad = quadratic_function_prime(x) # analytical gradient for x
num_grad = compute_numerical_gradient(quadratic_function, x) # numerical gradient for x
abs_error = np.abs(grad - num_grad) # absolute error between grad and num_grad

plot_gradients(grad, num_grad, 'Analytical', 'Numerical')

print('\nMean Absolute Error: {:.12e}'.format(np.mean(abs_error)))

