# import numpy as np
# from typing import Tuple
#
# radius = 8
# global_epsilon = 1e-9
#
#
# def differentiable_function(x: float, y: float) -> float:
#     return np.sin(x) * np.exp((1 - np.cos(y)) ** 2) + \
#         np.cos(y) * np.exp((1 - np.sin(x)) ** 2) + (x - y) ** 2
#
#
# def derivative_x(x: float, y: float) -> float:
#     return (differentiable_function(x + global_epsilon, y) - differentiable_function(x, y)) / global_epsilon
#
#
# def derivative_y(x: float, y: float) -> float:
#     return (differentiable_function(x, y + global_epsilon) - differentiable_function(x, y)) / global_epsilon
#
#
# def gradient_descent(initial_point: Tuple[float, float], method: str = 'armijo', max_iter: int = 1000, **kwargs) -> \
#         Tuple[float, float]:
#     x, y = initial_point
#     trajectory = [(x, y)]
#
#     for _ in range(max_iter):
#         dx = derivative_x(x, y)
#         dy = derivative_y(x, y)
#         grad = np.array([dx, dy])
#         direction = -grad
#
#         if method == 'armijo':
#             alpha = armijo_line_search(x, y, direction, **kwargs)
#         elif method == 'wolfe':
#             alpha = wolfe_line_search(x, y, direction, **kwargs)
#         elif method == 'golden':
#             alpha = golden_section_search(x, y, direction, **kwargs)
#         elif method == 'ternary':
#             alpha = ternary_line_search(x, y, direction, **kwargs)
#         else:
#             alpha = kwargs.get('learning_rate', 0.001)
#
#         x_new = x + alpha * direction[0]
#         y_new = y + alpha * direction[1]
#
#         if np.linalg.norm([x_new - x, y_new - y]) < global_epsilon:
#             break
#
#         x, y = x_new, y_new
#         trajectory.append((x, y))
#
#     return x, y
#
#
# def armijo_line_search(x: float, y: float, direction: np.ndarray[float],
#                        alpha_init: float = 1.0, c1: float = 1e-4,
#                        rho: float = 0.5, max_iters: int = 10) -> float:
#     alpha = alpha_init
#     f_current = differentiable_function(x, y)
#     grad = np.array([derivative_x(x, y), derivative_y(x, y)])
#     slope = c1 * np.dot(grad, direction)
#
#     for _ in range(max_iters):
#         x_new: float = float(x + alpha * direction[0])
#         y_new: float = float(y + alpha * direction[1])
#         f_new = differentiable_function(x_new, y_new)
#
#         if f_new <= f_current + alpha * slope:
#             return alpha
#         alpha *= rho
#
#     return alpha_init * (rho ** max_iters)
#
#
# def wolfe_line_search(x: float, y: float, direction: np.ndarray[float],
#                       alpha_init: float = 1.0, c1: float = 1e-4,
#                       c2: float = 0.9, max_iters: int = 20) -> float:
#     alpha = alpha_init
#     f_current = differentiable_function(x, y)
#     grad_current = np.array([derivative_x(x, y), derivative_y(x, y)])
#     slope = c1 * np.dot(grad_current, direction)
#
#     for _ in range(max_iters):
#         x_new: float = float(x + alpha * direction[0])
#         y_new: float = float(y + alpha * direction[1])
#         f_new = differentiable_function(x_new, y_new)
#         grad_new = np.array([derivative_x(x_new, y_new), derivative_y(x_new, y_new)])
#
#         if f_new > f_current + alpha * slope:
#             alpha *= 0.5
#             continue
#
#         if np.dot(grad_new, direction) < c2 * np.dot(grad_current, direction):
#             alpha *= 1.5
#             continue
#
#         return alpha
#
#     return alpha
#
#
# def golden_section_search(x: float, y: float, direction: np.ndarray,
#                           epsilon: float = 1e-6, max_iters: int = 100) -> float:
#     phi = (1 + np.sqrt(5)) / 2
#     resphi = 2 - phi
#
#     a = -1
#     b = 1
#
#     for _ in range(max_iters):
#         x1 = a + resphi * (b - a)
#         x2 = b - resphi * (b - a)
#
#         f1 = differentiable_function(x + x1 * direction[0], y + x1 * direction[1])
#         f2 = differentiable_function(x + x2 * direction[0], y + x2 * direction[1])
#
#         if f1 < f2:
#             b = x2
#         else:
#             a = x1
#
#         if abs(b - a) < epsilon:
#             break
#
#     return (a + b) / 2
#
#
# def ternary_line_search(x: float, y: float, direction: np.ndarray[float],
#                         epsilon: float = 1e-6, max_iters: int = 100) -> float:
#     left = 0
#     right = 1
#
#     def f(alpha: float) -> float:
#         return differentiable_function(float(x + alpha * direction[0]), float(y + alpha * direction[1]))
#
#     for _ in range(max_iters):
#         if abs(right - left) < epsilon:
#             break
#
#         m1 = left + (right - left) / 3
#         m2 = right - (right - left) / 3
#
#         f1 = f(m1)
#         f2 = f(m2)
#
#         if f1 < f2:
#             right = m2
#         else:
#             left = m1
#
#     return (left + right) / 2
#
#
# def find_minimum(initial_point: Tuple[float, float]) -> Tuple[float, float]:
#     return gradient_descent(initial_point, method='golden')
