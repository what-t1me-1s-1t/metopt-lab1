import numpy as np
from .shared import differentiable_function, derivative_x, derivative_y, global_epsilon


def wolfe_line_search(x, y, alpha_init=0.8, c1=1e-4, c2=0.5, max_iters=1000):
    alpha = alpha_init
    f_current = differentiable_function(x, y)
    grad = np.array([derivative_x(x, y), derivative_y(x, y)])
    direction = -grad
    derivative = np.dot(grad, direction)

    for _ in range(max_iters):
        x_new = x + alpha * direction[0]
        y_new = y + alpha * direction[1]
        f_new = differentiable_function(x_new, y_new)
        grad = np.array([derivative_x(x_new, y_new), derivative_y(x_new, y_new)])
        direction = -grad
        derivative_new = np.dot(grad, direction)

        if f_new > f_current + alpha * c1 * derivative:
            alpha *= 0.5
            continue

        if abs(derivative_new) >= c2 * abs(derivative):
            return alpha

        alpha *= 1.5

    return alpha