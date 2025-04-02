import numpy as np
from random import  uniform

global_epsilon = 1e-15


def differentiable_function(x: float, y: float, noise_left_bound: float = 0, noise_right_bound: float = 0) -> float:
    # return x ** 2 + y ** 2
    # return 0.26*(x**2 + y**2) - 0.48*x*y
    # return (1 + x)**2 +(y - x**2)**2
    # return 0.5*(x**2)+0.25*y**2-1
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2 + uniform(noise_left_bound, noise_right_bound)


def derivative_x(x: float, y: float) -> float:
    return (differentiable_function(x + global_epsilon, y) - differentiable_function(x - global_epsilon, y)) / (
            2 * global_epsilon)


def derivative_y(x: float, y: float) -> float:
    return (differentiable_function(x, y + global_epsilon) - differentiable_function(x, y - global_epsilon)) / (
            2 * global_epsilon)
