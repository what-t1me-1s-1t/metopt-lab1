from graphic import *
from math_logic import *

if __name__ == '__main__':
    initial_point = (4.0, 3.0)

    min_x, min_y = find_minimum(initial_point)
    minimum = (min_x, min_y, differentiable_function(min_x, min_y))

    grid = get_grid(0.05)
    draw_chart(minimum, grid)

    print(minimum)
