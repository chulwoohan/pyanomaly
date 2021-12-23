import numpy as np
from numba import njit


@njit(error_model='numpy')
def std(a, ddof=1):
    n = a.shape[0]
    s = a.std()
    return s if ddof == 0 else s * np.sqrt(n / (n - 1))


@njit(error_model='numpy')
def skew(a, ddof=1):
    n = a.shape[0]
    m = a.mean()
    s = std(a, ddof)
    a = (a - m) / s

    return (a ** 3).mean() if ddof == 0 else (a ** 3).sum() * n / (n-1) / (n-2)

@njit
def add_constant(a):
    return np.hstack((np.ones((a.shape[0], 1)), a))


@njit
def isnan1(data):
    """Equivalent to np.isnan(axis=1)"""
    isnan = np.full(data.shape[0], False)
    for j in range(data.shape[1]):
        isnan |= np.isnan(data[:, j])
    return isnan


