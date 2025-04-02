from grafics.graphic import *
from searches.main import gradient_descent
from searches.shared import differentiable_function

# 0.4, 0.2
# 2, 4
# 1, 1
# 4, 3
if __name__ == '__main__':
    initial_point = (3, 3)

    min_x, min_y, trajectory = gradient_descent(
        initial_point,
        method='armijo',
        noise_left_bound=-0.001,
        noise_right_bound=0.001,
        max_iter=1000
    )
    print(trajectory)
    minimum = (min_x, min_y, differentiable_function(min_x, min_y))

    grid = get_grid(0.05)
    draw_chart(minimum, grid)
    draw_xy_trajectory(trajectory)

    print(minimum)
