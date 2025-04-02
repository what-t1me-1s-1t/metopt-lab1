import numpy as np
from scipy.optimize import minimize

from grafics.graphic import get_grid, draw_chart, draw_xy_trajectory
from searches.main import gradient_descent
from searches.shared import differentiable_function, differentiable_function_sp

if __name__ == '__main__':
    initial_point = (0, 0)

    min_x, min_y, trajectory = gradient_descent(
        initial_point,
        noise_level=1e-11,
        method='armijo',
        max_iter=1000
    )

    print(trajectory)

    our_minimum = (min_x, min_y, differentiable_function(min_x, min_y))

    res_scipy = minimize(differentiable_function_sp, np.array(initial_point), method='BFGS', tol=1e-8)
    scipy_minimum = (res_scipy.x[0], res_scipy.x[1], res_scipy.fun)

    print("\n Результаты исследований\n")
    print(f"  Метод градиентного спуска (Wolfe):")
    print(f"   X = {our_minimum[0]:.10f}, Y = {our_minimum[1]:.10f}")
    print(f"   F(X, Y) = {our_minimum[2]:.10f}")
    print(f"   Итерации: {len(trajectory)}\n")

    print(f"  Метод Nelder-Mead (Scipy):")
    print(f"   X = {scipy_minimum[0]:.10f}, Y = {scipy_minimum[1]:.10f}")
    print(f"   F(X, Y) = {scipy_minimum[2]:.10f}")
    print(f"   Итерации: {res_scipy.nit}\n")

    methods = ["BFGS", "Nelder-Mead"]

    results = {}

    for method in methods:
        res = minimize(differentiable_function_sp, np.array(initial_point), method=method, tol=1e-8)
        results[method] = {
            "X": res.x[0],
            "Y": res.x[1],
            "F(X, Y)": res.fun,
            "Iterations": res.nit if "nit" in res else "N/A"
        }

    # Вывод результатов в красивом виде
    print("\n🔹 Сравнение методов оптимизации 🔹\n")
    for method, data in results.items():
        print(f"  Метод {method}:")
        print(f"   X = {data['X']:.6f}, Y = {data['Y']:.6f}")
        print(f"   F(X, Y) = {data['F(X, Y)']:.6f}")
        print(f"   Итерации: {data['Iterations']}\n")

    grid = get_grid(0.05)
    draw_chart(our_minimum, grid, initial_point)
    draw_xy_trajectory(trajectory)
