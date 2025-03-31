import numpy as np
from .shared import differentiable_function, derivative_x, derivative_y


def armijo_line_search(x: float, y: float, direction: np.ndarray[float],
                       alpha_init: float = 1.0, c1: float = 0.9,
                       rho: float = 0.09, max_iters: int = 10) -> float:
    alpha = alpha_init
    f_current = differentiable_function(x, y)
    grad = np.array([derivative_x(x, y), derivative_y(x, y)])
    slope = c1 * np.dot(grad, direction)

    for _ in range(max_iters):
        x_new: float = float(x + alpha * direction[0])
        y_new: float = float(y + alpha * direction[1])
        f_new = differentiable_function(x_new, y_new)

        if f_new <= f_current + alpha * slope:
            return alpha
        alpha *= rho

    return alpha_init * (rho ** max_iters)
