import numpy as np
from .shared import differentiable_function


def golden_section_search(x: float, y: float, direction: np.ndarray,
                          epsilon: float = 1e-6, max_iters: int = 1000) -> float:
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    a = -1
    b = 1

    for _ in range(max_iters):
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)

        f1 = differentiable_function(x + x1 * direction[0], y + x1 * direction[1])
        f2 = differentiable_function(x + x2 * direction[0], y + x2 * direction[1])

        if f1 < f2:
            b = x2
        else:
            a = x1

        if abs(b - a) < epsilon:
            break

    return (a + b) / 2
