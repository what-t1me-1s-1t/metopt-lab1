import matplotlib
import numpy as np

from math_logic import differentiable_function

matplotlib.use('Agg')
import matplotlib.pyplot as plt

radius = 8

def get_grid(grid_step):
    samples = np.arange(-radius, radius, grid_step)
    x, y = np.meshgrid(samples, samples)
    return x, y, differentiable_function(x, y)

def draw_chart(point, grid):
    point_x, point_y, point_z = point
    grid_x, grid_y, grid_z = grid

    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    angles = [30, 60, 90]
    for i, angle in enumerate(angles):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.8, edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.scatter(point_x, point_y, point_z, color='red', s=100, label='Minimum Point')

        ax.set_title(f'3D Surface Plot - Angle {angle}°', fontsize=12)
        ax.set_xlabel('X-axis', fontsize=10)
        ax.set_ylabel('Y-axis', fontsize=10)
        ax.set_zlabel('Function Value', fontsize=10)
        ax.legend()

        ax.view_init(elev=30, azim=angle)

        plt.tight_layout()
        plt.savefig(f'./graphics/gradient_descent_chart_{i}.png')

def draw_xy_trajectory(trajectory):
    plt.figure(figsize=(6, 6), dpi=300)
    print(len(trajectory))

    trajectory = np.array(trajectory)

    plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=5, linewidth=2, label='Gradient Descent Path')
    plt.title('Gradient Descent Trajectory in XY Plane')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('./graphics/xy_trajectory.png')

