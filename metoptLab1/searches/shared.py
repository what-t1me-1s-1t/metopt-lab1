import numpy as np

global_epsilon = 1e-15


def differentiable_function(x: float, y: float) -> float:
    # return np.sin(x) * np.exp((1 - np.cos(y)) ** 2) + \
    #     np.cos(y) * np.exp((1 - np.sin(x)) ** 2) + (x - y) ** 2
    return x ** 2 + y ** 2


def derivative_x(x: float, y: float) -> float:
    return (differentiable_function(x + global_epsilon, y) - differentiable_function(x, y)) / global_epsilon


def derivative_y(x: float, y: float) -> float:
    return (differentiable_function(x, y + global_epsilon) - differentiable_function(x, y)) / global_epsilon
