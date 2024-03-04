"""This module defines jitted function.

    .. autosummary::
        :nosignatures:

        nansum
        nanmean
        nanvar
        nanstd
        nanskew
        shift
        roll_sum
        roll_mean
        roll_var
        roll_std
        rank
        set_to_nan
        isnan1
        add_constant
        bivariate_regression
        regression
        rolling_regression
"""
# CODING TIPS
#
# Returning different types doesn't work. For example, returning a 1D or 2D array depending on a condition fails.

import os
import numpy as np
import pandas as pd
import numba
from numba import njit, prange, jit_module


def is_jitted(function):
    return isinstance(function, numba.core.registry.CPUDispatcher)


@njit(error_model='numpy', cache=True)
def nansum(x):
    """Sum excluding nan.

    Args:
        x: 1D ndarray.

    Returns:
        Sum of `x` excluding nan.
    """

    return np.nansum(x)


@njit(error_model='numpy', cache=True)
def nanmean(x):
    """Mean excluding nan.

    Args:
        x: 1D ndarray.

    Returns:
        Mean of `x` excluding nan.
    """

    return np.nanmean(x)


@njit(error_model='numpy', cache=True)
def nanstd(x, dof=1):
    """Standard deviation excluding nan.

    Args:
        x: 1D ndarray.
        dof: Degrees of freedom. 0 (1) for biased (unbiased) estimate. Default to 1.

    Returns:
        Standard deviation of `x` excluding nan.
    """

    n = x.shape[0] - np.isnan(x).sum()
    s = np.nanstd(x)
    return s if dof == 0 else s * np.sqrt(n / (n - dof))


@njit(error_model='numpy', cache=True)
def nanvar(x, dof=1):
    """Variance excluding nan.

    Args:
        x: 1D ndarray.
        dof: Degrees of freedom. 0 (1) for biased (unbiased) estimate. Default to 1.

    Returns:
        Variance of `x` excluding nan.
    """

    return nanstd(x, dof)**2


@njit(error_model='numpy', cache=True)
def nanskew(x, dof=1):
    """Skewness excluding nan.

    Args:
        x: 1D ndarray.
        dof: Degrees of freedom. 0 (1) for biased (unbiased) estimate. Default to 1.

    Returns:
        Skewness of `x` excluding nan.
    """

    n = x.shape[0] - np.isnan(x).sum()
    m = np.nanmean(x)
    s = nanstd(x, dof)
    x = (x - m) / s

    return (x ** 3).mean() if dof == 0 else (x ** 3).sum() * n / (n - 1) / (n - 2)


@njit(error_model='numpy', cache=True)
def shift(x, n):
    """Shift.

    `x` is shifted by `n` along the first axis. A negative `n` shifts `x` backward.

    Args:
        x: Ndarray.
        n: Shift size.

    Returns:
        Shifted `x`. Ndarray of the same size as `x`.
    """

    if x.ndim == 1:
        x_shift = np.roll(x, n)
    else:
        x_shift = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            x_shift[:, j] = np.roll(x[:, j], n)

    if n > 0:
        x_shift[:n] = np.nan
    else:
        x_shift[n:] = np.nan

    return x_shift


@njit(error_model='numpy', cache=True)
def roll_sum(x, n, min_n=-1):
    """Rolling sum.

    The `x` is rolled along the first axis with the window size `n`, and the sum of each window is calculated.
    Nan values are excluded: The result will be nan if the number of not-nan values is smaller than `min_n`.

    Args:
        x: Ndarray.
        n: Window size.
        min_n: Minimum number of observations. Default to `n`.

    Returns:
        Rolling sum. Ndarray of the same size as `x`.

    Examples:
        >>> x = np.array([[1, 2, 0, np.nan, 3, -1], [1, 2, 3, 4, 5, 6]]).T
        ... x
        array([[ 1.,  1.],
               [ 2.,  2.],
               [ 0.,  3.],
               [nan,  4.],
               [ 3.,  5.],
               [-1.,  6.]])

        >>> roll_sum(x, 3)
        array([[nan, nan],
               [nan, nan],
               [ 3.,  6.],
               [nan,  9.],
               [nan, 12.],
               [nan, 15.]])

        >>> roll_sum(x, 3, 2)
        array([[nan, nan],
               [ 3.,  3.],
               [ 3.,  6.],
               [ 2.,  9.],
               [ 3., 12.],
               [ 2., 15.]])
    """

    def sum1d(x):
        sum = np.nancumsum(x)
        sum[n:] = sum[n:] - sum[:-n]
        n_obs = np.nancumsum(~np.isnan(x))  # Number of not-Nan values in the window.
        n_obs[n:] = n_obs[n:] - n_obs[:-n]
        sum[n_obs < min_n] = np.nan
        return sum

    min_n = n if min_n < 0 else min_n
    res = np.full_like(x, np.nan)

    if x.shape[0] < min_n:
        return res

    if x.ndim == 1:
        res[:] = sum1d(x)
    else:
        for j in range(x.shape[1]):
            res[:, j] = sum1d(x[:, j])

    return res


@njit(error_model='numpy', cache=True)
def roll_mean(x, n, min_n=-1):
    """Rolling mean.

    The `x` is rolled along the first axis with the window size `n`, and the mean of each window is calculated.
    Nan values are excluded: The result will be nan if the number of not-nan values is smaller than `min_n`.

    Args:
        x: Ndarray.
        n: Window size.
        min_n: Minimum number of observations. Default to `n`.

    Returns:
        Rolling mean. Ndarray of the same size as `x`.
    """

    def mean1d(x):
        sum = np.nancumsum(x)
        sum[n:] = sum[n:] - sum[:-n]
        n_obs = np.nancumsum(~np.isnan(x))  # Number of not-Nan values in the window.
        n_obs[n:] = n_obs[n:] - n_obs[:-n]
        sum = sum / n_obs
        sum[n_obs < min_n] = np.nan
        return sum

    min_n = n if min_n < 0 else min_n
    res = np.full_like(x, np.nan)

    if x.shape[0] < min_n:
        return res

    if x.ndim == 1:
        res[:] = mean1d(x)
    else:
        for j in range(x.shape[1]):
            res[:, j] = mean1d(x[:, j])

    return res


@njit(error_model='numpy', cache=True)
def roll_var(x, n, min_n=-1, dof=1):
    """Rolling variance.

    The `x` is rolled along the first axis with the window size `n`, and the variance of each window is calculated.
    Nan values are excluded: The result will be nan if the number of not-nan values is smaller than `min_n`.

    Args:
        x: Ndarray.
        n: Window size.
        min_n: Minimum number of observations. Default to `n`.
        dof: Degrees of freedom. 0 (1) for biased (unbiased) estimate. Default to 1.

    Returns:
        Rolling variance. Ndarray of the same size as `x`.
    """

    def n_obs1d(x):
        n_obs = np.nancumsum(~np.isnan(x))  # Number of not-Nan values in the window.
        n_obs[n:] = n_obs[n:] - n_obs[:-n]
        return n_obs

    min_n = n if min_n < 0 else min_n
    res = np.full_like(x, np.nan)

    if x.shape[0] < min_n:
        return res

    mean = roll_mean(x, n, min_n)
    mean2 = roll_mean(x ** 2, n, min_n)
    var = mean2 - mean ** 2
    res[:] = np.where(var < 0., 0., var)  # Negative variance can occur due to numerical precision.

    if dof == 0:
        return res

    if x.ndim == 1:
        d = n_obs1d(x)
        res[:] *= d / (d - dof)
    else:
        for j in range(x.shape[1]):
            d = n_obs1d(x[:, j])
            res[:, j] *= d / (d - dof)

    return res


@njit(error_model='numpy', cache=True)
def roll_std(x, n, min_n=-1, dof=1):
    """Rolling standard deviation.

    The `x` is rolled along the first axis with the window size `n`, and the standard deviation of each window is
    calculated. Nan values are excluded: The result will be nan if the number of not-nan values is smaller than `min_n`.

    Args:
        x: Ndarray.
        n: Window size.
        min_n: Minimum number of observations. Default to `n`.
        dof: Degrees of freedom. 0 (1) for biased (unbiased) estimate. Default to 1.

    Returns:
        Rolling standard deviation. Ndarray of the same size as `x`.
    """

    return np.sqrt(roll_var(x, n, min_n, dof))


@njit(cache=True)
def rank(x, ascending=True, pct=False):
    """Rank.

    Calculate ranks of the elements of `x` within each column. Nan values are excluded.

    Args:
        x: Ndarray.
        ascending: If True, rank increases with value starting from 1.
        pct: If True, percentile ranks are returned.

    Returns:
        (Percentile) ranks of `x`. Ndarray of the same size as `x`.

    Examples:
        >>> x = np.array([[1, 2, 0, np.nan, 3, -1], [1, 2, 3, 4, 5, 6]]).T
        ... x
        array([[ 1.,  1.],
               [ 2.,  2.],
               [ 0.,  3.],
               [nan,  4.],
               [ 3.,  5.],
               [-1.,  6.]])

        >>> rank(x)
        array([[ 3.,  1.],
               [ 4.,  2.],
               [ 2.,  3.],
               [nan,  4.],
               [ 5.,  5.],
               [ 1.,  6.]])

        >>> rank(x, pct=True)
        array([[0.6       , 0.16666667],
               [0.8       , 0.33333333],
               [0.4       , 0.5       ],
               [       nan, 0.66666667],
               [1.        , 0.83333333],
               [0.2       , 1.        ]])
    """

    def rank1d(x):
        isna = np.isnan(x)
        x1 = x[~isna]

        idx = np.argsort(x1)
        if not ascending:
            idx = idx[::-1]

        rank = np.full(x1.shape, np.nan)
        rank[idx] = np.arange(x1.shape[0]) + 1.0

        s = x1[idx]
        dup = s[:-1][s[1:] == s[:-1]]
        for d in dup:
            a = x1 == d
            rank[a] = rank[a].mean()

        if pct:
            rank /= x1.shape[0]

        nanrank = np.full(x.shape, np.nan)
        nanrank[~isna] = rank
        return nanrank

    res = np.full_like(x, np.nan)

    if x.ndim == 1:
        res[:] = rank1d(x)
    else:
        for j in range(x.shape[1]):
            res[:, j] = rank1d(x[:, j])

    return res


@njit(cache=True)
def set_to_nan(x, n):
    """Set rows to nan.

    Set the first `n` (last `-n` if `n` < 0) rows of `x` to nan. The `x` is changed in-place.

    Args:
        x: Ndarray.
        n: Number of rows to set to nan.
    """

    if n > 0:
        x[:n] = np.nan
    else:
        x[n:] = np.nan


@njit(cache=True)
def isnan1(x):
    """Check nan along columns.

    Args:
        x: NxK ndarray.

    Returns:
        Nx1 bool ndarray. True if any value in the corresponding row of `x` is nan, False otherwise.

    Examples:
        >>> x = np.array([[np.nan, 1, 2], [1, 2, 3]])
        ... x
        [[nan  1.  2.]
         [ 1.  2.  3.]]

        >>> isnan1(x)
        [ True, False]
    """

    if x.ndim == 1:
        return np.isnan(x)
    else:
        isnan = np.full(x.shape[0], False)
        for j in range(x.shape[1]):
            isnan |= np.isnan(x[:, j])
        return isnan


@njit(cache=True)
def add_constant(x):
    """Add a constant column to a matrix.

    A vector of 1's is prepended to `x`.

    Args:
        x: N x K ndarray.

    Returns:
        N x (K+1) ndarray (`x` with a vector of 1's prepended).

    Examples:
        >>> x = np.array([[1, 2], [3, 4]])
        ... x
        [[1 2]
         [3 4]]
        >>> add_constant(x)
        [[1 1 2]
         [1 3 4]]
    """

    return np.hstack((np.ones((x.shape[0], 1), dtype=x.dtype), x))


@njit(error_model='numpy', cache=True)
def bivariate_regression(y, x):
    """Bivariate regression.

    A constant is added internally.

    Args:
        y: Nx1 ndarray. Dependent variable.
        x: Nx1 ndarray. Independent variable.

    Returns:
        * Coefficients. 2x1 ndarray of [constant, beta].
        * R-squared.
        * Residuals. Nx1 ndarray.
    """

    cov = np.cov(y, x)
    beta = cov[0][1] / cov[1][1]
    c = y.mean() - beta * x.mean()
    res = y - c - beta * x
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()

    return np.array([c, beta]), r2, res


@njit(error_model='numpy', cache=True)
def regression(y, X):
    """Multivariate regression.

    Args:
        y: Nx1 ndarray. Dependent variable.
        X: NxK ndarray. Independent variables (including constant).

    Returns:
        * Coefficients. Kx1 ndarray.
        * R-squared.
        * Residuals. Nx1 ndarray.
    """

    beta = (np.linalg.solve(X.T @ X, X.T @ y)).ravel()
    res = y - (beta * X).sum(1)
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()

    return beta, r2, res


@njit(cache=True)
def rolling_regression(data, n, min_n=-1, add_const=True):
    """Rolling regression.

    The `data` is rolled along the first axis with the window size `n`, and the regression is conducted for each window.
    Nan values are excluded: The result will be nan if the number of not-nan values is smaller than `min_n`.


    Args:
        data: NxK ndarray. The first column is the dependent variable and the rest are the independent variables.
        n: Window size.
        min_n: Minimum number of observations. Default to `n`.
        add_const: If True, add a constant to the independent variables.

    Returns:
        Nx(K'+2) ndarray. K' = K if add_const is True, else K' = K-1.

            * First K' columns: Coefficients.
            * K'+1-th column: R-squared.
            * K'+2-th column: Idiosyncratic volatility (standard deviation of residuals).
    """

    min_n = n if min_n < 0 else min_n
    nobs = data.shape[0]
    nx = data.shape[1]
    retval = np.full((nobs, nx + 2), np.nan, dtype=data.dtype)

    if nobs < min_n:
        return retval

    not_nan = ~isnan1(data)
    Y = data[:, 0]
    X = data[:, 1:]
    if add_const:
        X = add_constant(X)

    for i in range(min_n - 1, nobs):
        s, e = i - min(i, n - 1), i + 1

        not_na = not_nan[s:e]
        y = Y[s:e][not_na]
        x = X[s:e][not_na]

        if y.shape[0] < min_n:
            continue

        try:
            beta, r2, res = regression(y, x)
        except:
            continue

        retval[i, :nx] = beta
        retval[i, -2] = r2
        retval[i, -1] = nanstd(res)

    return retval


def clear_cache():
    dir = './pyanomaly/__pycache__/'
    files = os.listdir(dir)
    for file in files:
        path = dir + file
        os.remove(path)



# jit_module(nopython=True, error_model='numpy')


# @njit
# def _get_X_n_y_rolled(
#     a: np.ndarray, window: int
# ) -> tuple[np.ndarray, np.ndarray]:
#     y = a[:, 0]
#     X = a[:, 1:]
#     X = add_constant(X)
#     X_rolled = np.lib.stride_tricks.sliding_window_view(X, window, axis=0)
#     y_rolled = np.lib.stride_tricks.sliding_window_view(y, window, axis=0)
#     X_rolled = X_rolled.transpose(0, 2, 1)
#     return X_rolled, y_rolled
#
#
# def _get_beta_vec_from_Xy(
#     X_rolled: np.ndarray, y_rolled: np.ndarray
# ) -> np.ndarray:
#     XT_rolled = np.transpose(X_rolled, axes=(0, 2, 1))
#     XT_X_rolled = np.einsum("tik,tkj -> tij", XT_rolled, X_rolled)
#     Xy_rolled = np.einsum("tik,tk -> ti",  XT_rolled, y_rolled)
#     return np.linalg.solve(XT_X_rolled, Xy_rolled)
#
# def get_beta_vectorized(
#     a: np.ndarray, window: int
# ) -> np.ndarray:
#     X_rolled, y_rolled = _get_X_n_y_rolled(a, window)
#     return _get_beta_vec_from_Xy(X_rolled, y_rolled)
#
# @njit
# def get_beta_hat(
#     X: np.ndarray, y: np.ndarray
# ) -> float:
#     return (np.linalg.solve(X.T @ X, X.T @ y))
#
# @njit
# def get_rolling_func_loop(
#     a: np.ndarray, window: int
# ) -> np.ndarray:
#     X_rolled, y_rolled = _get_X_n_y_rolled(a, window)
#
#     beta = np.full_like(X_rolled, np.nan)
#     for t in range(y_rolled.shape[0]):
#         X = X_rolled[t]
#         y = y_rolled[t]
#         beta[t] = get_beta_hat(X, y)
#     return beta
#
#


if __name__ == '__main__':
    os.chdir('../')
    # clear_cache()

    # pct = True
    x = np.array([[1, 2, 0, np.nan,  3, -1], [1, 2, 3, 4, 5, 6]]).T
    roll_sum(x, 2)
    # print(rank(x, ascending=False, pct=pct))
    # print(pd.DataFrame(x).rank(ascending=False, pct=pct))
    # s = np.sort(x[:, 1], axis=None)
    # print(s[:-1][s[1:] == s[:-1]])

