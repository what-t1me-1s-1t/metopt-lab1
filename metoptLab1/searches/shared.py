import numpy as np

global_epsilon = 1e-8


def differentiable_function(x: float, y: float, noise_level: float = 0) -> float:
    # return x ** 2 + y ** 2
    # return 0.26*(x**2 + y**2) - 0.48*x*y
    # return (1 + x)**2 +(y - x**2)**2
    # return 0.5*(x**2)+0.25*y**2-1
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2 + np.random.normal(0, noise_level)

def differentiable_function_sp(x):
    x_val, y_val = x[0], x[1]
    return differentiable_function(x_val, y_val)


def derivative_x(x: float, y: float) -> float:
    return (differentiable_function(x + global_epsilon, y) - differentiable_function(x - global_epsilon, y)) / (
            2 * global_epsilon)


def derivative_y(x: float, y: float) -> float:
    return (differentiable_function(x, y + global_epsilon) - differentiable_function(x, y - global_epsilon)) / (
            2 * global_epsilon)
