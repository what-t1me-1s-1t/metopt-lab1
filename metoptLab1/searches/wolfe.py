import numpy as np
from .shared import differentiable_function, derivative_x, derivative_y, global_epsilon


def wolfe_line_search(x: float, y: float, direction: np.ndarray[float],
                      alpha_init: float = 1.0, c1: float = 1e-4,
                      c2: float = 0.9, max_iters: int = 20) -> float:
    alpha = alpha_init
    f_current = differentiable_function(x, y)
    grad_current = np.array([derivative_x(x, y), derivative_y(x, y)])
    slope = c1 * np.dot(grad_current, direction)

    for _ in range(max_iters):
        x_new: float = float(x + alpha * direction[0])
        y_new: float = float(y + alpha * direction[1])
        f_new = differentiable_function(x_new, y_new)
        grad_new = np.array([derivative_x(x_new, y_new), derivative_y(x_new, y_new)])

        if f_new > f_current + alpha * slope:
            alpha *= 0.5
            continue

        if np.dot(grad_new, direction) < c2 * np.dot(grad_current, direction):
            alpha *= 1.5
            continue

        return alpha

    return alpha
