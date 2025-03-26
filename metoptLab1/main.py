from graphics.graphic import *
from math_logic import *

if __name__ == '__main__':
    initial_point = (2, 2)

    min_x, min_y, trajectory = find_minimum(initial_point)
    minimum = (min_x, min_y, differentiable_function(min_x, min_y))

    grid = get_grid(0.05)
    draw_chart(minimum, grid)
    draw_xy_trajectory(trajectory)
    print(minimum)
