import numpy as np

radius = 8
global_epsilon = 1e-9
centre = (global_epsilon, global_epsilon)
arr_shape = 100
step = radius / arr_shape





# def calculate_flip_points():
#     flip_points = np.array([0, 0])
#     points = np.zeros((360, arr_shape), dtype=bool)
#     cx, cy = centre
#
#     for i in range(arr_shape):
#         for alpha in range(360):
#             x, y = rotate_vector(step, alpha)
#             x = x * i + cx
#             y = y * i + cy
#             points[alpha][i] = derivative_x(x, y) + derivative_y(x, y) > 0
#             if i > 0 and not points[alpha][i - 1] and points[alpha][i]:
#                 flip_points = np.vstack((flip_points, np.array([alpha, i - 1])))
#
#     return flip_points

#
# def pick_estimates(positions):
#     if len(positions) < 2:
#         return centre
#
#     vx, vy = rotate_vector(step, positions[1][0])
#     cx, cy = centre
#     best_x, best_y = cx + vx * positions[1][1], cy + vy * positions[1][1]
#
#     for index in range(2, len(positions)):
#         vx, vy = rotate_vector(step, positions[index][0])
#         x, y = cx + vx * positions[index][1], cy + vy * positions[index][1]
#         if differentiable_function(best_x, best_y) > differentiable_function(x, y):
#             best_x, best_y = x, y
#
#     for index in range(360):
#         vx, vy = rotate_vector(step, index)
#         x, y = cx + vx * (arr_shape - 1), cy + vy * (arr_shape - 1)
#         if differentiable_function(best_x, best_y) > differentiable_function(x, y):
#             best_x, best_y = x, y
#
#     return best_x, best_y
#


def differentiable_function(x, y):
    return np.sin(x) * np.exp((1 - np.cos(y)) ** 2) + \
        np.cos(y) * np.exp((1 - np.sin(x)) ** 2) + (x - y) ** 2


def rotate_vector(length, a):
    return length * np.cos(a), length * np.sin(a)


def derivative_x(x, y):
    return (differentiable_function(x + global_epsilon, y) - differentiable_function(x, y)) / global_epsilon


def derivative_y(x, y):
    return (differentiable_function(x, y + global_epsilon) - differentiable_function(x, y)) / global_epsilon


def gradient_descent(initial_point, method='armijo', max_iter=1000, **kwargs):
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
        else:
            alpha = kwargs.get('learning_rate', 1e-9)

        x_new = x + alpha * direction[0]
        y_new = y + alpha * direction[1]

        if np.linalg.norm([x_new - x, y_new - y]) < global_epsilon:
            break

        x, y = x_new, y_new
        trajectory.append((x, y))

    return x, y


def armijo_line_search(x, y, direction, alpha_init=1.0, c1=1e-4, rho=0.5, max_iters=10):
    alpha = alpha_init
    f_current = differentiable_function(x, y)
    grad = np.array([derivative_x(x, y), derivative_y(x, y)])
    slope = c1 * np.dot(grad, direction)

    for _ in range(max_iters):
        x_new = x + alpha * direction[0]
        y_new = y + alpha * direction[1]
        f_new = differentiable_function(x_new, y_new)

        if f_new <= f_current + alpha * slope:
            return alpha
        alpha *= rho

    return alpha_init * (rho ** max_iters)


def wolfe_line_search(x, y, direction, alpha_init=1.0, c1=1e-4, c2=0.9, max_iters=20):
    alpha = alpha_init
    f_current = differentiable_function(x, y)
    grad_current = np.array([derivative_x(x, y), derivative_y(x, y)])
    slope = c1 * np.dot(grad_current, direction)

    for _ in range(max_iters):
        x_new = x + alpha * direction[0]
        y_new = y + alpha * direction[1]
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


def find_minimum(initial_point):
    return gradient_descent(initial_point, method='wolfe')



