# centre = (global_epsilon, global_epsilon)
# arr_shape = 100
# step = radius / arr_shape





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



# def rotate_vector(length, a):
#     return length * np.cos(a), length * np.sin(a)