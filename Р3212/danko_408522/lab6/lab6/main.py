import math
import matplotlib.pyplot as plt

def function_1(x, y):
    """
    Функция: y' = y
    Точное решение: y(x) = e^x
    """
    return y

def function_2(x, y):
    """
    Функция: y' = x
    Точное решение: y(x) = x^2 / 2
    """
    return x

def function_3(x, y):
    """
    Функция: y' = sin(x)
    Точное решение: y(x) = 1 - cos(x)
    """
    return math.sin(x)

def exact_solution_1(x):
    """
    Точное решение для y' = y
    """
    return math.exp(x)

def exact_solution_2(x):
    """
    Точное решение для y' = x
    """
    return (x ** 2) / 2

def exact_solution_3(x):
    """
    Точное решение для y' = sin(x)
    """
    return 1 - math.cos(x)

# Функция выбора
def select_function(choice):
    if choice == 1:
        return function_1, exact_solution_1, "y' = y"
    elif choice == 2:
        return function_2, exact_solution_2, "y' = x"
    elif choice == 3:
        return function_3, exact_solution_3, "y' = sin(x)"
    else:
        raise ValueError("Некорректный выбор функции.")

def print_table(points, exact_solution):
    """
    Выводит таблицу значений с точным и приближенным решением.

    Параметры:
    points - список точек (x, y)
    exact_solution - функция точного решения
    """
    print(f"{'x':>10} {'y (численное)':>20} {'y (точное)':>20} {'Ошибка':>20}")
    print("=" * 70)
    for x, y in points:
        y_exact = exact_solution(x)
        error = abs(y - y_exact)
        print(f"{x:>10.4f} {y:>20.6f} {y_exact:>20.6f} {error:>20.6e}")

def get_user_input():
    """
    Запрашивает у пользователя начальные условия и параметры для численного решения ОДУ.
    Возвращает кортеж (x0, xn, h, epsilon).
    """
    try:
        print("Введите параметры для численного решения задачи Коши:")

        x0 = float(input("Начальное значение x (x0): "))
        xn = float(input("Конечное значение x (xn): "))

        if xn <= x0:
            raise ValueError("Конечное значение должно быть больше начального.")

        h = float(input("Шаг интегрирования (h): "))

        if h <= 0:
            raise ValueError("Шаг интегрирования должен быть положительным.")

        epsilon = float(input("Точность (epsilon): "))

        if epsilon <= 0:
            raise ValueError("Точность должна быть положительным числом.")

        print(f"\nПараметры успешно введены:\n x0 = {x0}\n xn = {xn}\n h = {h}\n epsilon = {epsilon}")
        return x0, xn, h, epsilon

    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        return get_user_input()

def calculate_error(exact_values, approx_values):
    """
    Вычисляет максимальную ошибку между точными и приближенными значениями.

    Параметры:
    exact_values  - список точных значений y_i
    approx_values - список приближенных значений y_i

    Возвращает:
    Максимальное значение ошибки.
    """
    if len(exact_values) != len(approx_values):
        raise ValueError("Длины списков точных и приближенных значений должны совпадать.")

    max_error = max(abs(e - a) for e, a in zip(exact_values, approx_values))
    return max_error

def runge_criteria(y_h, y_h2):
    """
    Вычисляет отношение погрешностей по критерию Рунге.
    """
    return abs(y_h2 - y_h) / (2**2 - 1)

def compare_methods(f, exact_solution, x0, y0, xn, h):
    """
    Сравнивает численные методы с шагами h и h/2.

    Параметры:
    f - функция правой части уравнения dy/dx = f(x, y)
    exact_solution - точная функция для сравнения
    x0 - начальное значение x
    y0 - начальное значение y
    xn - конечное значение x
    h - шаг интегрирования
    """
    methods = [
        ("Euler", euler_method),
        ("Improved Euler", improved_euler_method),
        ("Adams", adams_method)
    ]

    print(f"\n=== Сравнение методов при h = {h} и h/2 = {h / 2} ===")

    for name, method in methods:
        # Решения с шагами h и h/2
        result_h = method(f, x0, y0, xn, h)
        result_h2 = method(f, x0, y0, xn, h / 2)

        # Извлечение значений x и y
        x_h, y_h = zip(*result_h)
        x_h2, y_h2 = zip(*result_h2)

        # Уменьшаем список точек с h/2, чтобы он соответствовал списку с шагом h
        x_h2 = x_h2[::2]
        y_h2 = y_h2[::2]

        # Точные значения для обоих шагов
        exact_values_h = [exact_solution(x) for x in x_h]
        exact_values_h2 = [exact_solution(x) for x in x_h2]

        # Вычисление ошибок
        error_h = calculate_error(exact_values_h, y_h)
        error_h2 = calculate_error(exact_values_h2, y_h2)

        # Критерий Рунге
        runge_estimates = [
            runge_criteria(y1, y2) for y1, y2 in zip(y_h, y_h2)
        ]
        max_runge = max(runge_estimates) if runge_estimates else 0.0

        print(f"\nМетод: {name}")
        print(f"Ошибка при h: {error_h:.6e}")
        print(f"Ошибка при h/2: {error_h2:.6e}")
        print(f"Максимальное значение по критерию Рунге: {max_runge:.6e}")



def plot_solution(exact_solution, approx_points, method_name, file):
    """
    Строит графики точной функции и приближенного решения.

    Параметры:
    exact_solution - функция точного решения
    approx_points  - список точек (x, y) приближенного решения
    method_name    - название метода для отображения на графике
    """
    # Извлечение координат из точек
    x_approx, y_approx = zip(*approx_points)

    # Построение точной функции на тех же точках x
    y_exact = [exact_solution(x) for x in x_approx]

    # Построение графика
    plt.figure(figsize=(8, 5))
    plt.plot(x_approx, y_exact, label="Точное решение", linestyle='-', color='blue', marker='o')
    plt.plot(x_approx, y_approx, label=f"Приближенное решение ({method_name})", linestyle='--', color='red', marker='x')

    # Оформление графика
    plt.title(f"Сравнение точного и приближенного решений методом {method_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig(file)


def euler_method(f, x0, y0, xn, h):
    """
    Метод Эйлера для численного решения задачи Коши для ОДУ первого порядка.

    Параметры:
    f  - функция правой части уравнения dy/dx = f(x, y)
    x0 - начальное значение x
    y0 - начальное значение y (y(x0))
    xn - конечное значение x
    h  - шаг интегрирования

    Возвращает:
    Список точек (x, y), представляющих приближенное решение.
    """
    # Начальные условия
    x = x0
    y = y0
    points = [(x, y)]

    # Итерация по методу Эйлера
    while x < xn:
        y = y + h * f(x, y)  # вычисляем y_{n+1}
        x = x + h            # вычисляем x_{n+1}
        points.append((x, y))

    return points

def improved_euler_method(f, x0, y0, xn, h):
    """
    Улучшенный метод Эйлера (метод Эйлера-Коши) для численного решения задачи Коши для ОДУ первого порядка.

    Параметры:
    f  - функция правой части уравнения dy/dx = f(x, y)
    x0 - начальное значение x
    y0 - начальное значение y (y(x0))
    xn - конечное значение x
    h  - шаг интегрирования

    Возвращает:
    Список точек (x, y), представляющих приближенное решение.
    """
    # Начальные условия
    x = x0
    y = y0
    points = [(x, y)]

    # Итерация по улучшенному методу Эйлера
    while x < xn:
        # Шаг 1: Вычисляем промежуточное значение (предсказатель)
        k1 = f(x, y)
        y_tilde = y + h * k1

        # Шаг 2: Корректируем значение (уточнитель)
        k2 = f(x + h, y_tilde)
        y = y + (h / 2) * (k1 + k2)
        x = x + h

        points.append((x, y))

    return points

def adams_method(f, x0, y0, xn, h):
    """
    Метод Адамса для численного решения задачи Коши для ОДУ первого порядка.

    Параметры:
    f  - функция правой части уравнения dy/dx = f(x, y)
    x0 - начальное значение x
    y0 - начальное значение y (y(x0))
    xn - конечное значение x
    h  - шаг интегрирования

    Возвращает:
    Список точек (x, y), представляющих приближенное решение.
    """
    # Начальные условия
    x = x0
    y = y0
    points = [(x, y)]

    # Вычисляем первую точку методом улучшенного Эйлера
    # Получаем список точек и берем последнюю
    improved_points = improved_euler_method(f, x, y, x + h, h)
    x1, y1 = improved_points[-1]  # Последняя точка из метода Эйлера
    points.append((x1, y1))

    # Инициализируем переменные для Адамса
    x_prev, y_prev = x, y
    x, y = x1, y1

    # Итерация по методу Адамса
    while x < xn:
        # Предсказание методом Адамса-Бэшфорта (2-го порядка)
        y_pred = y + (h / 2) * (3 * f(x, y) - f(x_prev, y_prev))

        # Уточнение методом Адамса-Мултона (2-го порядка)
        y_corr = y + (h / 2) * (f(x + h, y_pred) + f(x, y))

        # Обновляем точки
        x_prev, y_prev = x, y
        x = x + h
        y = y_corr

        points.append((x, y))

    return points


def main():
    print("=== Численное решение задачи Коши для ОДУ ===")
    print("Выберите функцию для задачи Коши:")
    print("1: y' = y (Точное решение: y(x) = e^x)")
    print("2: y' = x (Точное решение: y(x) = x^2 / 2)")
    print("3: y' = sin(x) (Точное решение: y(x) = 1 - cos(x))")

    try:
        choice = int(input("Ваш выбор (1/2/3): "))
        f, exact_solution, description = select_function(choice)
        print(f"\nВы выбрали: {description}")

        # Получаем начальные данные от пользователя
        x0, xn, h, epsilon = get_user_input()

        # Инициализируем начальное значение y0
        y0 = exact_solution(x0)

        # Вычисляем таблицы значений с помощью различных методов
        print("\n=== Таблицы значений численных методов ===")
        euler_points = euler_method(f, x0, y0, xn, h)
        improved_points = improved_euler_method(f, x0, y0, xn, h)
        adams_points = adams_method(f, x0, y0, xn, h)

        # Отображаем таблицы значений
        print("\nМетод Эйлера:")
        print_table(euler_points, exact_solution)

        print("\nУлучшенный метод Эйлера:")
        print_table(improved_points, exact_solution)

        print("\nМетод Адамса:")
        print_table(adams_points, exact_solution)

        # Сравниваем методы
        print("\n=== Сравнение численных методов ===")
        compare_methods(f, exact_solution, x0, y0, xn, h)

        # Построение графиков
        print("\n=== Построение графиков решений ===")
        plot_solution(exact_solution, euler_points, "Эйлера", "euler_method.png")
        plot_solution(exact_solution, improved_points, "Улучшенного Эйлера", "improved_euler_method.png")
        plot_solution(exact_solution, adams_points, "Адамса", "adams_method.png")

    except ValueError as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
