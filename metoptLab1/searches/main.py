import numpy as np
from typing import Tuple, Union, Any

from numpy import ndarray, dtype, unsignedinteger

from .shared import global_epsilon, derivative_x, derivative_y
from .armijo import armijo_line_search
from .wolfe import wolfe_line_search
from .golden_sieve import golden_section_search
from .ternary_search import ternary_line_search


def gradient_descent(initial_point: Tuple[float, float], method: str = 'armijo', max_iter: int = 1000, **kwargs) -> \
        tuple[Union[Union[float, np.ndarray[Any, np.dtype[np.unsignedinteger[Any]]]], Any], Union[float, Any], list[
            Union[tuple[float, float], tuple[Union[np.ndarray[Any, np.dtype[np.unsignedinteger[Any]]], Any], Any]]]]:
    x, y = initial_point
    trajectory = [(x, y)]

    for _ in range(max_iter):
        dx = derivative_x(x, y)
        dy = derivative_y(x, y)
        grad = np.array([dx, dy])
        direction = -grad

        if method == 'armijo':
            alpha = armijo_line_search(x, y, direction, **kwargs)
        elif method == 'wolfe':
            alpha = wolfe_line_search(x, y, direction, **kwargs)
        elif method == 'golden':
            alpha = golden_section_search(x, y, direction, **kwargs)
        elif method == 'ternary':
            alpha = ternary_line_search(x, y, direction, **kwargs)
        else:
            alpha = kwargs.get('learning_rate', 0.09)

        x_new = x + alpha * direction[0]
        y_new = y + alpha * direction[1]

        if np.linalg.norm([x_new - x, y_new - y]) < global_epsilon:
            break

        x, y = x_new, y_new
        trajectory.append((x, y))

    return x, y, trajectory


def find_minimum(initial_point: Tuple[float, float]) -> tuple[
    Union[Union[float, ndarray[Any, dtype[unsignedinteger[Any]]]], Any], Union[float, Any], list[
        Union[tuple[float, float], tuple[Union[ndarray[Any, dtype[unsignedinteger[Any]]], Any], Any]]]]:
    return gradient_descent(initial_point, method='golden')
