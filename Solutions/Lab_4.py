import random
import math
import numpy as np


def integral_1(x):
    return x * math.atan(x)


def integral_2(x):
    return (x ** 4 - 3 * x ** 2 + 1) * math.exp(-0.5 * x ** 2)


def integral_3(x, y):
    return math.sqrt((y + 1) * y + math.sin(x) ** 2) / (y - 0.7 * x * x + 0.3)


def monte_carlo_1d(func, a, b, n=10000):
    return (b - a) / n * np.sum([func(random.uniform(a, b)) for _ in np.arange(n)])


def area_g(x, y):
    if math.fabs(x) < 2 and 4 >= y >= x ** 2:
        return 1
    return -1


def monte_carlo_2d(f, g, x0, x1, y0, y1, n):
    x = np.random.uniform(x0, x1, n)
    y = np.random.uniform(y0, y1, n)
    f_mean = 0
    num_inside = 0
    for i in np.arange(n):
        for j in np.arange(n):
            if g(x[i], y[j]) >= 0:
                num_inside += 1
                f_mean += f(x[i], y[j])

    f_mean = f_mean / float(num_inside)
    area = num_inside / float(n ** 2) * (x1 - x0) * (y1 - y0)
    return area * f_mean


if __name__ == "__main__":
    mc_1 = monte_carlo_1d(integral_1, 0, math.sqrt(3))
    print("Monte Carlo: {}; Wolfram Alpha: {}".format(mc_1, 1.2284))
    mc_2 = monte_carlo_1d(integral_2, -50, 50)
    print("Monte Carlo: {}; Wolfram Alpha: {}".format(mc_2, math.sqrt(2 * math.pi)))
    mc_3 = monte_carlo_2d(integral_3, area_g, -2, 2, 0, 4, 1000)
    print("Monte Carlo: {}; Wolfram Alpha: {}".format(mc_3, 15.6951))
