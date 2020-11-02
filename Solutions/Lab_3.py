import numpy as np
import math
import random
import matplotlib.pyplot as plt


def normal_distribution(mu, sigma, n_factor=12, n=10000):
    distribution = np.zeros(n)
    sqrt_sigma = np.sqrt(sigma)
    for i in np.arange(n):
        pol = np.sum([np.random.rand() for i in np.arange(n_factor)]) - n_factor / 2
        distribution[i] = mu + sqrt_sigma * np.sqrt(12 / n_factor) * pol
    return distribution


def get_expected_value(sequence):
    return np.sum(sequence) / len(sequence)


def get_dispersion(sequence):
    expected_value = get_expected_value(sequence)
    n = len(sequence)
    return np.sum([sequence[i] ** 2 - expected_value ** 2 for i in np.arange(n)]) / (n - 1)


def exp_distribution(lambd, n=10000):
    distribution = np.zeros(n)
    for i in np.arange(n):
        a = np.random.rand()
        distribution[i] = (-1 / lambd) * math.log(a)
    return distribution


def get_chi_sq_value(k):
    distribution = [random.normalvariate(0, 1) for _ in np.arange(k)]
    return sum([distribution[i] ** 2 for i in np.arange(k)])


def fisher_distribution(l, m, n=10000):
    return [(get_chi_sq_value(l) / l) / (get_chi_sq_value(m) / m) for _ in np.arange(n)]


def box_muller_transform_normal_distribution(n=10000):
    is_ready = False
    second = 0.0
    mu = 0
    sigma = 1
    distribution = np.zeros(n)
    for i in np.arange(n):
        if is_ready:
            is_ready = False
            distribution[i] = mu + second * sigma
            continue
        else:
            while True:
                u = 2.0 * random.random() - 1.0
                v = 2.0 * random.random() - 1.0
                s = u * u + v * v
                if s <= 1.0 and s != 0.0:
                    r = np.sqrt(-2.0 * math.log(s) / s)
                    second = r * u
                    is_ready = True
                    distribution[i] = mu + r * v * sigma
                    break
    return distribution


def mix(siq_1, siq_2, pi):
    distribution = np.zeros((len(siq_1)))
    for i in np.arange(len(siq_1)):
        if random.random() < pi:
            distribution[i] = siq_1[i]
        else:
            distribution[i] = siq_2[i]
    return distribution


if __name__ == "__main__":
    dist_params = (0, 1)
    normal_sequence = normal_distribution(*dist_params)
    expected_value_normal = get_expected_value(normal_sequence)
    print("Unbiased expected value = {}, True expected value = {}".format(expected_value_normal, dist_params[0]))
    dispersion_normal = get_dispersion(normal_sequence)
    print("Unbiased dispersion value = {}, True dispersion = {}".format(dispersion_normal, dist_params[1]))
    exp_params = 0.25
    exp_sequence = exp_distribution(exp_params)
    expected_value_exp = get_expected_value(exp_sequence)
    print("Unbiased expected value = {}, True expected value = {}".format(expected_value_exp, 1 / exp_params))
    dispersion_exp = get_dispersion(exp_sequence)
    print("Unbiased dispersion value = {}, True dispersion = {}".format(dispersion_exp, 1 / exp_params ** 2))
    fisher_params = (5, 3)
    fisher_sequence = fisher_distribution(*fisher_params)
    expected_fisher = fisher_params[1] / (fisher_params[1] - 2)
    expected_value_fisher = get_expected_value(fisher_sequence)
    print("Unbiased expected value = {}, True expected value = {}".format(expected_value_fisher, expected_fisher))
    dispersion_fisher = get_dispersion(fisher_sequence)
    # rus -> dispersiya
    print("Unbiased dispersion value = {}, True dispersion = {}".format(dispersion_fisher, 0))
    bm_trans_normal = box_muller_transform_normal_distribution()
    print("Unbiased expected value = {}, True expected value = {}".format(get_expected_value(bm_trans_normal), 0))
    print("Unbiased dispersion value = {}, True dispersion = {}".format(get_dispersion(bm_trans_normal), 1))
    mix_params = 0.6
    mix_sequence = mix(exp_sequence, fisher_sequence, mix_params)
    mix_value_normal = get_expected_value(mix_sequence)
    print("Unbiased expected value = {}".format(expected_value_normal))
    dispersion_mix = get_dispersion(mix_sequence)
    print("Unbiased dispersion value = {}".format(dispersion_mix))
    hist, bins = np.histogram(bm_trans_normal, bins=1000)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
