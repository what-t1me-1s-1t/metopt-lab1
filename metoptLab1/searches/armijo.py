import numpy as np
from .shared import differentiable_function, derivative_x, derivative_y

def armijo_line_search(x, y, alpha_init=0.5, c1=1e-4, q=0.5, max_iters=1000):
    alpha = alpha_init
    f_current = differentiable_function(x, y)
    grad = np.array([derivative_x(x, y), derivative_y(x, y)])
    direction = -grad

    for _ in range(max_iters):
        x_new = x + alpha * direction[0]
        y_new = y + alpha * direction[1]

        f_new = differentiable_function(x_new, y_new)
        derivative = np.dot(grad, direction)

        if f_new < f_current + c1 * alpha * derivative:
            return alpha

        alpha = q * alpha

        grad = np.array([derivative_x(x_new, y_new), derivative_y(x_new, y_new)])
        direction = -grad

    return alpha

