import numpy as np
from .shared import differentiable_function, derivative_x, derivative_y


def armijo_line_search(x: float, y: float, direction: np.ndarray[float],
                       alpha_init: float = 1, c1: float = 0.1,
                       rho: float = 0.1, max_iters: int = 1000) -> float:
    grad_x = derivative_x(x, y)
    grad_y = derivative_y(x, y)
    grad = np.array([grad_x, grad_y])

    t = -c1 * np.dot(grad, direction)

    alpha = alpha_init
    for _ in range(max_iters):
        if differentiable_function(x, y) - differentiable_function(x + alpha * direction[0], y + alpha * direction[1]) >= alpha * t:
            return alpha
        alpha *= rho

    return alpha
