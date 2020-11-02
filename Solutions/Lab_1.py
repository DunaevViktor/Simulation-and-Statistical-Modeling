import matplotlib.pyplot as plt
import numpy as np


def mcm(beta, M, alpha, n=10000):
    sequence = np.zeros(n)
    prev_astrix_alpha = alpha
    for i in np.arange(n):
        current_astrix_alpha = (beta * prev_astrix_alpha) % M
        sequence[i] = current_astrix_alpha / M
        prev_astrix_alpha = current_astrix_alpha
    return sequence


def mmm(first_sequence, second_sequence, k, n=10000):
    v = np.zeros(k)
    result_sequence = np.zeros(n)
    for i in np.arange(k - 1):
        v[i] = first_sequence[i]
    for i in np.arange(n):
        s = int((second_sequence[i] * k) // 1)
        result_sequence[i] = v[s]
        v[s] = first_sequence[i]
    return result_sequence


def cd(sequence, delta):
    n = len(sequence)
    m = (1 / n) * sum(sequence)
    s_square = (1 / (n - 1)) * sum((sequence - m) ** 2)
    ksi_1 = m - 0.5
    ksi_2 = s_square - 1 / 12
    c_1 = (12 * n) ** 0.5
    c_2 = ((n - 1) / n) * (0.0056 * pow(n, -1) + 0.0028 * pow(n, -2) - 0.0083 * pow(n, -3)) ** -0.5
    h = (c_1 * abs(ksi_1) < delta) & (c_2 * abs(ksi_2) < delta)
    if h:
        print('Good sequence')
    else:
        print('Bad sequence')


def ct(sequence):
    n = len(sequence)
    # Вборочное среднее
    sm = 1 / 2 * sum(sequence)
    t = int(sm)
    r_j = np.zeros(t)
    # for j in np.arange(t):
    #     r_j[j] = 1/(n-j-1) * 


if __name__ == "__main__":
    zero_alpha = 445423
    c_one = 90474281
    M = 2 ** 31
    beta = max(c_one, M - c_one)
    sequence = mcm(beta, M, zero_alpha)
    print('Task one: ')
    print('1 element = {}'.format(sequence[0]))
    print('2 element = {}'.format(sequence[1]))
    print('15 element = {}'.format(sequence[14]))
    print('100 element = {}'.format(sequence[99]))
    print('900 element = {}'.format(sequence[899]))
    print('1000 element = {}'.format(sequence[999]))
    zero_alpha_2 = 39565801
    c_one_two = 123534789
    k = 192
    sequence_2 = mcm(max(c_one_two, M - c_one_two), M, zero_alpha_2)
    mmm_sequence = mmm(sequence, sequence_2, k)
    print('Task one: ')
    print('1 element = {}'.format(mmm_sequence[0]))
    print('15 element = {}'.format(mmm_sequence[14]))
    print('100 element = {}'.format(mmm_sequence[99]))
    print('900 element = {}'.format(mmm_sequence[899]))
    print('1000 element = {}'.format(mmm_sequence[999]))
    delta = 0.83 ** -1
    cd(sequence, delta)
    hist, bins = np.histogram(mmm_sequence, bins=10)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
