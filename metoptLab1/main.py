from graphic import *
from math_logic import *


if __name__ == '__main__':
    min_x, min_y = find_minimum()
    minimum = (min_x, min_y, differentiable_function(min_x, min_y))
    grid = get_grid(0.05)
    draw_chart(minimum, grid)
    print(minimum)