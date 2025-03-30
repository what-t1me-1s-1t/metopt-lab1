import numpy as np
from .shared import differentiable_function


def ternary_line_search(x: float, y: float, direction: np.ndarray[float],
                        epsilon: float = 1e-6, max_iters: int = 100) -> float:
    left = 0
    right = 1

    def f(alpha: float) -> float:
        return differentiable_function(float(x + alpha * direction[0]), float(y + alpha * direction[1]))

    for _ in range(max_iters):
        if abs(right - left) < epsilon:
            break

        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3

        f1 = f(m1)
        f2 = f(m2)

        if f1 < f2:
            right = m2
        else:
            left = m1

    return (left + right) / 2
