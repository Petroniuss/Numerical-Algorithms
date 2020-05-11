from cmath import exp, pi
import numpy as np


def dft_matrix(n):
    Fn = np.empty(shape=(n, n), dtype=np.complex64)
    for j in range(n):
        for k in range(n):
            Fn[j, k] = np.exp(-2 * np.pi * 1j * k * j / n)

    return Fn


def dft(x):
    x = x.reshape((-1, 1))
    Fn = dft_matrix(x.shape[0])

    return Fn @ x


def idft(X):
    X = X.reshape((-1, 1))
    n = X.shape[0]
    Fn = dft_matrix(n)

    return np.conjugate(Fn @ np.conjugate(X)) / n


def fft(x):
    n = len(x)
    if n <= 1:
        return x

    m = n // 2
    evenBin = np.empty(m)
    oddBin = np.empty(m)
    for i in range(m):
        evenBin[i] = x[2 * i]
        oddBin[i] = x[2 * i + 1]

    evenF = fft(evenBin)
    oddF = fft(oddBin)

    X = np.empty(n, dtype=np.complex64)
    for k in range(m):
        exp = np.exp((-2 * np.pi * k * 1j / n))
        pf = exp * oddF[k]

        X[k] = evenF[k] + pf
        X[k + m] = evenF[k] - pf

    return X
