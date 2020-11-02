import numpy as np
import random as rng


def bernulli(p):
    n = 1000
    ber = np.zeros(2)
    for _ in np.arange(n):
        brv = rng.random()
        if brv <= p:
            ber[1] += 1
        else:
            ber[0] += 1
    expected_value = ber[1] / n
    variance = ((0 - expected_value) ** 2) * ber[0] / n + ((1 - expected_value) ** 2) * ber[1] / n
    print('Expected value = {}, True expected value = {}'.format(expected_value, p))
    print('Variance = {}, True variance = {}'.format(variance, p * (1 - p)))


def binomical(m, p):
    n = 1000
    bin = np.zeros(m)
    for _ in np.arange(n):
        a = np.random.rand(m)
        x = sum([0 if (p - a[i]) <= 0 else 1 for i in np.arange(m)])
        bin[x - 1] += 1
    expected_value = sum([i * bin[i] / n for i in np.arange(m)])
    variance = sum([((i - expected_value) ** 2) * bin[i] / n for i in np.arange(m)])
    print('Expected value = {}, True expected value = {}'.format(expected_value, m * p))
    print('Variance = {}, True variance = {}'.format(variance, m * p * (1 - p)))


if __name__ == "__main__":
    bernulli(0.2)
    binomical(6, 0.75)
