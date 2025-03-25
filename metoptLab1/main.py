import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple


class StepSizeStrategy(ABC):
    """Абстрактный класс для стратегии выбора шага"""

    @abstractmethod
    def get_step_size(self, f: Callable, x: np.ndarray, grad: np.ndarray,
                      direction: np.ndarray) -> float:
        """
        Вычисляет размер шага для текущей итерации

        Параметры:
            f: целевая функция
            x: текущая точка
            grad: градиент в текущей точке
            direction: направление спуска

        Возвращает:
            Размер шага (alpha)
        """
        pass


class ArmijoStepSize(StepSizeStrategy):
    """Реализация правила Армихо для выбора шага"""

    def __init__(self, c: float = 1e-4, beta: float = 0.5, max_iter: int = 100):
        """
        Инициализация параметров правила Армихо

        Параметры:
            c: параметр достаточного убывания (обычно 1e-4)
            beta: коэффициент уменьшения шага (обычно 0.1-0.8)
            max_iter: максимальное число итераций для поиска шага
        """
        self.c = c
        self.beta = beta
        self.max_iter = max_iter

    def get_step_size(self, f: Callable, x: np.ndarray,
                      grad: np.ndarray, direction: np.ndarray) -> float:
        """
        Вычисляет шаг по правилу Армихо

        Условие Армихо: f(x + αd) ≤ f(x) + c*α*grad(f)^T*d
        где d - направление спуска
        """
        alpha = 1.0  # Начальное значение шага
        current_value = f(x)
        directional_derivative = grad.T @ direction

        for _ in range(self.max_iter):
            new_x = x + alpha * direction
            new_value = f(new_x)

            # Проверка условия Армихо
            if new_value <= current_value + self.c * alpha * directional_derivative:
                return alpha

            alpha *= self.beta  # Уменьшаем шаг

        return alpha  # Возвращаем последнее значение, если не сошлось


class GradientDescent:
    """Класс градиентного спуска с возможностью выбора стратегии шага"""

    def __init__(self, step_strategy: StepSizeStrategy = ArmijoStepSize(),
                 tol: float = 1e-6, max_iter: int = 1000):
        """
        Инициализация метода градиентного спуска

        Параметры:
            step_strategy: стратегия выбора шага (по умолчанию Armijo)
            tol: критерий остановки (по умолчанию 1e-6)
            max_iter: максимальное число итераций (по умолчанию 1000)
        """
        self.step_strategy = step_strategy
        self.tol = tol
        self.max_iter = max_iter
        self.history = []

    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray,
                 verbose: bool = False) -> Tuple[np.ndarray, float, int]:
        """
        Выполняет оптимизацию методом градиентного спуска

        Параметры:
            f: целевая функция
            grad_f: функция вычисления градиента
            x0: начальная точка
            verbose: флаг вывода информации о процессе

        Возвращает:
            x: найденный минимум
            f(x): значение функции в минимуме
            n_iter: число выполненных итераций
        """
        x = x0.copy()
        self.history = [x.copy()]

        for n_iter in range(1, self.max_iter + 1):
            grad = grad_f(x)
            direction = -grad  # Направление градиентного спуска

            # Проверка критерия остановки
            if np.linalg.norm(grad) < self.tol:
                if verbose:
                    print(f"Оптимизация завершена за {n_iter} итераций")
                return x, f(x), n_iter

            # Выбор шага по заданной стратегии
            alpha = self.step_strategy.get_step_size(f, x, grad, direction)

            # Обновление точки
            x = x + alpha * direction
            self.history.append(x.copy())

            if verbose and n_iter % 100 == 0:
                print(f"Iter {n_iter}: f(x) = {f(x):.6f}, |grad| = {np.linalg.norm(grad):.6f}")

        if verbose:
            print(f"Достигнуто максимальное число итераций {self.max_iter}")
        return x, f(x), self.max_iter


# Пример использования
if __name__ == "__main__":
    # Тестовая квадратичная функция: f(x) = x^2 + y^2
    def quadratic(x):
        return x[0] ** 2 + x[1] ** 2


    def quadratic_grad(x):
        return np.array([2 * x[0], 2 * x[1]])


    # Начальная точка
    x0 = np.array([10.0, 5.0])

    # Создаем оптимизатор с правилом Армихо
    optimizer = GradientDescent(step_strategy=ArmijoStepSize(), verbose=True)

    # Запускаем оптимизацию
    x_opt, f_opt, n_iter = optimizer.optimize(quadratic, quadratic_grad, x0, verbose=True)

    print("\nРезультаты оптимизации:")
    print(f"Найденный минимум: {x_opt}")
    print(f"Значение функции: {f_opt:.6f}")
    print(f"Число итераций: {n_iter}")