import numpy as np
from typing import Tuple, Union, Any

from .shared import derivative_x, derivative_y
from .armijo import armijo_line_search
from .wolfe import wolfe_line_search
from .golden_sieve import golden_section_search
from .ternary_search import ternary_line_search


def gradient_descent(
        initial_point: Tuple[float, float],
        method: str,
        max_iter: int,
        noise_level: float = 0,
        **kwargs
) -> tuple[
    Union[Union[float, np.ndarray[Any, np.dtype[np.unsignedinteger[Any]]]], Any],
    Union[float, Any],
    list[Union[tuple[float, float], tuple[Union[np.ndarray[Any, np.dtype[np.unsignedinteger[Any]]], Any], Any]]]
]:
    x, y = initial_point
    trajectory = [(x, y)]
    eps = 1e-6

    for _ in range(max_iter):
        dx = derivative_x(x, y)
        dy = derivative_y(x, y)
        grad = np.array([dx, dy])
        direction = -grad

        if method == 'armijo':
            alpha = armijo_line_search(x, y, noise_level, **kwargs)
        elif method == 'wolfe':
            alpha = wolfe_line_search(x, y, **kwargs)
        elif method == 'golden':
            alpha = golden_section_search(x, y, direction, **kwargs)
        elif method == 'ternary':
            alpha = ternary_line_search(x, y, direction, **kwargs)
        else:
            alpha = kwargs.get('learning_rate', 3)

        x_new = x + alpha * direction[0]
        y_new = y + alpha * direction[1]

        if np.linalg.norm([x_new - x, y_new - y]) < eps:
            break

        x, y = x_new, y_new
        trajectory.append((x, y))

    return x, y, trajectory
