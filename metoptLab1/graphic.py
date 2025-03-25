import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plot
from math_logic import find_minimum,  differentiable_function


radius = 8

def get_grid(grid_step):
    samples = np.arange(-radius, radius, grid_step)
    x, y = np.meshgrid(samples, samples)
    return x, y, differentiable_function(x, y)

def draw_chart(point, grid):
    point_x, point_y, point_z = point
    grid_x, grid_y, grid_z = grid
    plot.rcParams.update({
        'figure.figsize': (4, 4),
        'figure.dpi': 200,
        'xtick.labelsize': 4,
        'ytick.labelsize': 4
    })
    ax = plot.figure().add_subplot(111, projection='3d')
    ax.scatter(point_x, point_y, point_z, color='red')
    ax.plot_surface(grid_x, grid_y, grid_z, rstride=5, cstride=5, alpha=0.7)
    plot.savefig('chart.png')


# if __name__ == '__main__':
#     min_x, min_y = find_minimum()
#     minimum = (min_x, min_y, differentiable_function(min_x, min_y))
#     grid = get_grid(0.05)
#     draw_chart(minimum, grid)
#     print(minimum)
