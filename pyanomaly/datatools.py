"""This module defines functions for data handling.

**Group-and-Apply**

    .. autosummary::
        :nosignatures:

        apply_to_groups
        apply_to_groups_jit
        apply_to_groups_reduce_jit

**Classify/Trim/Filter/Winsorize**

    .. autosummary::
        :nosignatures:

        classify
        trim
        filter
        winsorize

**Merge/Populate/Shift**

    .. autosummary::
        :nosignatures:

        merge
        populate
        shift

**Data Inspection/Comparison**

    .. autosummary::
        :nosignatures:

        inspect_data
        compare_data

**Auxiliary Functions**

    .. autosummary::
        :nosignatures:

        to_month_end
        add_months
"""

import numpy as np
import pandas as pd
from numba.typed import List
from pandas.tseries.offsets import MonthEnd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from pyanomaly.globals import *
from pyanomaly.numba_support import *


################################################################################
#
# GROUP-LEVEL APPLICATION OF A FUNCTION
#
################################################################################

def _parse_retval(retval, row_idx=None):
    """Aggregate returns from a groupby operation.

    The `retval` is a list of returns from groups. If an item of `retval` is a tuple (multiple returns), the return
    value of this function is a tuple of aggregated returns.

    Args:
        retval: List of one of the followings:
            i) (tuple of) float
            ii) (tuple of) ndarray
            iii) (tuple of) dataframe/series.
        row_idx: List of row index. If given, the aggregated values are reordered by `row_idx`.

    Returns:
        Concatenated value of `retval`. The number of returns is equal to the length of `retval[0]`.
    """

    if isinstance(retval[0], np.ndarray):
        retval = np.concatenate(retval)
        if (row_idx is None) or (retval.shape[0] != row_idx.shape[0]):
            return retval
        else:
            return retval[row_idx]
    elif isinstance(retval[0], (pd.Series, pd.DataFrame)):
        retval = pd.concat(retval)
        if (row_idx is None) or (retval.shape[0] != row_idx.shape[0]):
            return retval
        else:
            return retval.iloc[row_idx]
    elif is_float(retval[0]):
        return np.array(retval).reshape((-1, 1))
    elif isinstance(retval[0], tuple):  # multiple returns
        retvals = []
        for j in range(len(retval[0])):
            res_j = [res[j] for res in retval]

            if is_float(res_j[0]):
                res_j = np.array(res_j)
            if isinstance(res_j[0], np.ndarray):
                res_j = np.concatenate(res_j)
                if row_idx:
                    res_j = res_j[row_idx]
            else:
                res_j = pd.concat(res_j)
                if row_idx:
                    res_j = res_j.iloc[row_idx]
            retvals.append(res_j)

        return tuple(retvals)


def _parse_group_info(data, ginfo):
    """Parse grouping information."""

    if isinstance(ginfo, (int, str)) and isinstance(data, np.ndarray):
        raise ValueError('If grouping information is an integer or string, data must be a pandas Series or DataFrame.')

    if isinstance(ginfo, int):  # index level
        ginfo = data.groupby(level=ginfo)
    elif isinstance(ginfo, str):  # column or index
        ginfo = data.groupby(ginfo)

    if isinstance(ginfo, (pd.core.groupby.generic.SeriesGroupBy, pd.core.groupby.generic.DataFrameGroupBy)):
        # GroupBy object
        gidx = list(ginfo.indices.values())
        gidx_array = np.concatenate(gidx)
        if np.all(gidx_array[:-1] <= gidx_array[1:]):  # If indexes are sorted, use group sizes for faster performance.
            ginfo = ginfo.size().to_numpy()
        else:
            ginfo = gidx

    return ginfo


def _make_input_jit_compatible(data, data2=None, *args):
    # Convert list or tuple arguments to ndarray.
    args = tuple(np.array(arg) if isinstance(arg, (list, tuple)) else arg for arg in args)

    # Convert pandas Series or DataFrame to numpy array.
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.to_numpy()
    if (data2 is not None) and isinstance(data2, (pd.Series, pd.DataFrame)):
        data2 = data2.to_numpy()

    return data, data2, args


def _apply_to_groups_gsize(x, gsize, function, *args, y=None):
    """A sub-function of ``apply_to_groups``. This function is used when group sizes are given.
    """

    idx = np.full(1 + gsize.shape[0], 0)
    idx[1:] = np.cumsum(gsize)

    if isinstance(x, np.ndarray):
        if y is None:
            retval = [function(x[idx[i]:idx[i + 1]], *args) for i in range(gsize.shape[0])]
        else:
            retval = [function(x[idx[i]:idx[i + 1]], y[idx[i]:idx[i + 1]], *args) for i in range(gsize.shape[0])]
    else:
        if y is None:
            retval = [function(x.iloc[idx[i]:idx[i + 1]], *args) for i in range(gsize.shape[0])]
        else:
            retval = [function(x.iloc[idx[i]:idx[i + 1]], y.iloc[idx[i]:idx[i + 1]], *args) for i in
                      range(gsize.shape[0])]

    return _parse_retval(retval)


def _apply_to_groups_gidx(x, gidx, function, *args, y=None):
    """A sub-function of ``apply_to_groups``. This function is used when group indexes are given.
    """

    if isinstance(x, np.ndarray):
        if y is None:
            retval = [function(x[gidx[i]], *args) for i in range(len(gidx))]
        else:
            retval = [function(x[gidx[i]], y[gidx[i]], *args) for i in range(len(gidx))]
    else:
        if y is None:
            retval = [function(x.iloc[gidx[i]], *args) for i in range(len(gidx))]
        else:
            retval = [function(x.iloc[gidx[i]], y.iloc[gidx[i]], *args) for i in range(len(gidx))]

    return _parse_retval(retval, np.argsort(np.concatenate(gidx)))


def apply_to_groups(data, ginfo, function, *args, data2=None):
    """Group data and apply a function to each group.

    This function can be used for a complex groupby operation. The `data` (and `data2`) is grouped using the grouping
    information, `ginfo`, and `function` is applied to each group. The `function` can be either jitted or unjitted.
    If it is jitted, consider using :func:`apply_to_groups_jit` instead, which runs the for loop along the groups in
    parallel. This function is faster than ``groupby().apply(function)`` when the size of `data` is large.

    Args:
        data: DataFrame or ndarray (values of a DataFrame) to be grouped.
        ginfo: Grouping information: integer (index level), str (column name), pandas GroupBy object, ndarray of
            group sizes, or list of group indexes. See the note below.
        function: Function to apply to groups. Its arguments should be (`gbdata`, `*args`) or
            (`gbdata`, `gbdata2`, `*args`), where `gbdata` (`gbdata2`) is a group of `data` (`data2`).
        *args: Additional arguments of `function`.
        data2: DataFrame or ndarray (values of a DataFrame) to be grouped. Optional argument when `function` requires
            two sets of input data.

    Returns:
        Concatenated value of the outputs of `function`.

    Note:
        Suppose `data` is a DataFrame with index = date/id, sorted on id/date.

        To apply a function to each id, `ginfo` can be set to any of the followings.

        * ginfo = 'id' (index name)
        * ginfo = 1 (index level)
        * ginfo = data.groupby('id') (GroupBy object)
        * ginfo = data.groupby('id').size().to_numpy() (group size)
        * ginfo = list(data.groupby('id').indices.values()) (group index)

        To apply a function to each date, `ginfo` can be set to any of the followings.

        * ginfo = 'date' (index name)
        * ginfo = 0 (index level)
        * ginfo = data.groupby('date') (GroupBy object)
        * ginfo = list(data.groupby('date').indices.values()) (group index)

        Since `data` is sorted on id/date, group sizes can be used only when grouped by id. The most efficient method
        is to provide group sizes, followed by group indexes. Hence, if this function needs to be called repeatedly,
        performance can be improved by generating group sizes (if data is sorted on the grouping index (column))
        or group indexes outside and use them as `ginfo`.

    Examples:
        >>> data = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
        ...      'me_nyse': [np.nan, 5000, 2000, 300, 150, np.nan, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... ).sort_index(level=['permno', 'date'])
        ... data
                             ret    me  me_nyse
        date       permno
        2023-03-31 10000   0.010   100      NaN
        2023-04-30 10000  -0.030   120      NaN
        2023-03-31 20000   0.050  5000 5000.000
        2023-04-30 20000  -0.010  4500 4500.000
        2023-03-31 30000   0.020  2000 2000.000
        2023-04-30 30000   0.030  2100 2100.000
        2023-03-31 40000   0.030   300  300.000
        2023-04-30 40000  -0.050   305  305.000
        2023-03-31 50000   0.110   150  150.000
        2023-04-30 50000   0.070   140  140.000

        Define the function to apply.

        >>> def rolling_sum(x, n):
        ...    return x.rolling(n).sum()

        Group data by 'permno' and calculate rolling sum of 'ret' and 'me'.

        >>> apply_to_groups(data[['ret', 'me']], 'permno', rolling_sum, 2)
                             ret       me
        date       permno                
        2023-03-31 10000     NaN      NaN
        2023-04-30 10000  -0.020  220.000
        2023-03-31 20000     NaN      NaN
        2023-04-30 20000   0.040 9500.000
        2023-03-31 30000     NaN      NaN
        2023-04-30 30000   0.050 4100.000
        2023-03-31 40000     NaN      NaN
        2023-04-30 40000  -0.020  605.000
        2023-03-31 50000     NaN      NaN
        2023-04-30 50000   0.180  290.000

        The followings are equivalent to the above.
        
        >>> gb = data.groupby('permno')
        ... apply_to_groups(data[['ret', 'me']], gb, rolling_sum, 2)

        >>> gsize = gb.size().to_numpy()
        ... apply_to_groups(data[['ret', 'me']], gsize, rolling_sum, 2)

        Group data by 'date' and calculate cross-sectional mean of 'ret'.
        
        >>> apply_to_groups(data['ret'], 'date', np.mean)
        [[0.044]
         [0.002]]

        The followings are equivalent to the above.

        >>> gb = data.groupby('date')
        ... apply_to_groups(data['ret'], gb, np.mean)
         
        >>> gidx = list(gb.indices.values())
        ... apply_to_groups(data['ret'], gidx, np.mean)
    """

    ginfo = _parse_group_info(data, ginfo)

    if is_jitted(function):
        data, data2, args = _make_input_jit_compatible(data, data2, *args)

    if isinstance(ginfo, np.ndarray):  # group size
        return _apply_to_groups_gsize(data, ginfo, function, *args, y=data2)
    else:
        return _apply_to_groups_gidx(data, ginfo, function, *args, y=data2)
    # return retval.ravel() if n_ret == 1 else retval


@njit(parallel=config.jit_parallel)
def _apply_to_groups_gsize_jit_1(x, gsize, function, retval, *args):
    """A sub-function of ``apply_to_groups_jit``. This function is used when group sizes are given and `function`
    takes only one data input, `x`.
    """

    idx = np.full(1 + gsize.shape[0], 0)
    idx[1:] = np.cumsum(gsize)

    for i in prange(gsize.shape[0]):
        retval[idx[i]:idx[i + 1]] = function(x[idx[i]:idx[i + 1]], *args)

    return retval


@njit(parallel=config.jit_parallel)
def _apply_to_groups_gsize_jit_2(x, y, gsize, function, retval, *args):
    """A sub-function of ``apply_to_groups_jit``. This function is used when group sizes are given and `function`
    takes only two data inputs, `x` and `y`.
    """

    idx = np.full(1 + gsize.shape[0], 0)
    idx[1:] = np.cumsum(gsize)

    for i in prange(gsize.shape[0]):
        retval[idx[i]:idx[i + 1]] = function(x[idx[i]:idx[i + 1]], y[idx[i]:idx[i + 1]], *args)

    return retval


@njit(parallel=config.jit_parallel)
def _apply_to_groups_gidx_jit_1(x, gidx, function, retval, *args):
    """A sub-function of ``apply_to_groups_jit``. This function is used when group indexes are given and `function`
    takes only one data input, `x`.
    """

    for i in prange(len(gidx)):
        retval[gidx[i]] = function(x[gidx[i]], *args)

    return retval


@njit(parallel=config.jit_parallel)
def _apply_to_groups_gidx_jit_2(x, y, gidx, function, retval, *args):
    """A sub-function of ``apply_to_groups_jit``. This function is used when group indexes are given and `function`
    takes only two data inputs, `x` and `y`.
    """

    for i in prange(len(gidx)):
        retval[gidx[i]] = function(x[gidx[i]], y[gidx[i]], *args)

    return retval


def apply_to_groups_jit(data, ginfo, function, n_ret, *args, data2=None):
    """Group data and apply a function to each group (jitted version).

    This function is similar to :func:`apply_to_groups`. The for loop along the groups is jitted for faster
    performance. The first call of this function can be slow as jitting takes place when first called.
    The `function` should be jitted, and the row size of its returns should be the same as the row size of
    the input data. For reduce functions, e.g., sum and mean, use :func:`apply_to_groups_reduce_jit`.

    Args:
        data: DataFrame or ndarray (values of a DataFrame) to be grouped.
        ginfo: Grouping information: integer (index level), str (column name), pandas GroupBy object, ndarray of
            group sizes, or list of group indexes. See :func:`apply_to_groups` for more details.
        function: Jitted function to apply to groups. Its arguments should be (`gbdata`, `*args`) or
            (`gbdata`, `gbdata2`, `*args`), where `gbdata` (`gbdata2`) is a group of `data` (`data2`).
        n_ret: Number of returns of `function`. If None, it is assumed to be the same as the column size of `data`.
        *args: Additional arguments of `function`.
        data2: DataFrame or ndarray (values of a DataFrame) to be grouped. Optional argument when `function` requires
            two sets of input data.

    Returns:
        Concatenated value of the outputs of `function`.

    Examples:
        >>> data = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
        ...      'me_nyse': [np.nan, 5000, 2000, 300, 150, np.nan, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... ).sort_index(level=['permno', 'date'])
        ... data
                             ret    me  me_nyse
        date       permno
        2023-03-31 10000   0.010   100      NaN
        2023-04-30 10000  -0.030   120      NaN
        2023-03-31 20000   0.050  5000 5000.000
        2023-04-30 20000  -0.010  4500 4500.000
        2023-03-31 30000   0.020  2000 2000.000
        2023-04-30 30000   0.030  2100 2100.000
        2023-03-31 40000   0.030   300  300.000
        2023-04-30 40000  -0.050   305  305.000
        2023-03-31 50000   0.110   150  150.000
        2023-04-30 50000   0.070   140  140.000

        Group data by 'permno' and calculate rolling sum of 'ret' and 'me' using
        :func:`.numba_support.roll_sum`.

        >>> apply_to_groups_jit(data[['ret', 'me']], 'permno', roll_sum, 2, 2)
        array([[      nan,       nan],
               [-2.00e-02,  2.20e+02],
               [      nan,       nan],
               [ 4.00e-02,  9.50e+03],
               [      nan,       nan],
               [ 5.00e-02,  4.10e+03],
               [      nan,       nan],
               [-2.00e-02,  6.05e+02],
               [      nan,       nan],
               [ 1.80e-01,  2.90e+02]])

        The followings are equivalent to the above.

        >>> gb = data.groupby('permno')
        ... apply_to_groups_jit(data[['ret', 'me']], gb, roll_sum, 2, 2)

        >>> gsize = gb.size().to_numpy()
        ... apply_to_groups_jit(data[['ret', 'me']], gsize, roll_sum, 2, 2)
    """

    if config.disable_jit:
        return apply_to_groups(data, ginfo, function, *args, data2=data2)

    ginfo = _parse_group_info(data, ginfo)
    data, data2, args = _make_input_jit_compatible(data, data2, *args)

    if not n_ret:
        retval = np.full_like(data, np.nan)
    elif n_ret == 1:
        retval = np.full((data.shape[0],), np.nan, dtype=data.dtype)
    else:
        retval = np.full((data.shape[0], n_ret), np.nan, dtype=data.dtype)

    if isinstance(ginfo, np.ndarray):  # group size
        if data2 is None:
            _apply_to_groups_gsize_jit_1(data, ginfo, function, retval, *args)
        else:
            _apply_to_groups_gsize_jit_2(data, data2, ginfo, function, retval, *args)
    elif isinstance(ginfo, list):  # group index list
        if data2 is None:
            _apply_to_groups_gidx_jit_1(data, List(ginfo), function, retval, *args)
        else:
            _apply_to_groups_gidx_jit_2(data, data2, List(ginfo), function, retval, *args)

    return retval.ravel() if n_ret == 1 else retval


@njit(parallel=config.jit_parallel)
def _apply_to_groups_reduce_gsize_jit_1(x, gsize, function, retval, *args):
    """A sub-function of ``apply_to_groups_jit``. This function is used when group sizes are given and `function`
    takes only one data input, `x`.
    """

    idx = np.full(1 + gsize.shape[0], 0)
    idx[1:] = np.cumsum(gsize)

    for i in prange(gsize.shape[0]):
        retval[i] = function(x[idx[i]:idx[i + 1]], *args)

    return retval


@njit(parallel=config.jit_parallel)
def _apply_to_groups_reduce_gsize_jit_2(x, y, gsize, function, retval, *args):
    """A sub-function of ``apply_to_groups_reduce_jit``. This function is used when group sizes are given and `function`
    takes only two data inputs, `x` and `y`.
    """

    idx = np.full(1 + gsize.shape[0], 0)
    idx[1:] = np.cumsum(gsize)

    for i in prange(gsize.shape[0]):
        retval[i] = function(x[idx[i]:idx[i + 1]], y[idx[i]:idx[i + 1]], *args)

    return retval


@njit(parallel=config.jit_parallel)
def _apply_to_groups_reduce_gidx_jit_1(x, gidx, function, retval, *args):
    """A sub-function of ``apply_to_groups_reduce_jit``. This function is used when group indexes are given and
    `function` takes only one data input, `x`.
    """

    for i in prange(len(gidx)):
        retval[i] = function(x[gidx[i]], *args)

    return retval


@njit(parallel=config.jit_parallel)
def _apply_to_groups_reduce_gidx_jit_2(x, y, gidx, function, retval, *args):
    """A sub-function of ``apply_to_groups_reduce_jit``. This function is used when group indexes are given and
    `function` takes only two data inputs, `x` and `y`.
    """

    for i in prange(len(gidx)):
        retval[i] = function(x[gidx[i]], y[gidx[i]], *args)

    return retval


def apply_to_groups_reduce_jit(data, ginfo, function, n_ret, *args, data2=None):
    """Group data and apply a reduce function to each group (jitted version).

    This function is similar to :func:`apply_to_groups`. The for loop along the groups is jitted for faster
    performance. The first call of this function can be slow as jitting takes place when first called.
    The `function` should be jitted and a reduce function such as mean or std: the row size of the returns should be 1.

    Args:
        data: DataFrame or ndarray (values of a DataFrame) to be grouped.
        ginfo: Grouping information: integer (index level), str (column name), pandas GroupBy object, ndarray of
            group sizes, or list of group indexes. See :func:`apply_to_groups` for more details.
        function: Jitted function to apply to groups. Its arguments should be (`gbdata`, `*args`) or
            (`gbdata`, `gbdata2`, `*args`), where `gbdata` (`gbdata2`) is a group of `data` (`data2`).
        n_ret: Number of returns of `function`. If None, it is assumed to be the same as the column size of `data`.
        *args: Additional arguments of `function`.
        data2: DataFrame or ndarray (values of a DataFrame) to be grouped. Optional argument when `function` requires
            two sets of input data.

    Returns:
        Concatenated value of the outputs of `function`.

    Examples:
        >>> data = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
        ...      'me_nyse': [np.nan, 5000, 2000, 300, 150, np.nan, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... ).sort_index(level=['permno', 'date'])
        ... data
                             ret    me  me_nyse
        date       permno
        2023-03-31 10000   0.010   100      NaN
        2023-04-30 10000  -0.030   120      NaN
        2023-03-31 20000   0.050  5000 5000.000
        2023-04-30 20000  -0.010  4500 4500.000
        2023-03-31 30000   0.020  2000 2000.000
        2023-04-30 30000   0.030  2100 2100.000
        2023-03-31 40000   0.030   300  300.000
        2023-04-30 40000  -0.050   305  305.000
        2023-03-31 50000   0.110   150  150.000
        2023-04-30 50000   0.070   140  140.000

        Group data by 'date' and calculate cross-sectional standard deviation of 'ret' using
        :func:`.numba_support.nanstd`.

        >>> apply_to_groups_reduce_jit(data['ret'], 'date', nanstd, None)
        array([0.03974921, 0.04816638])

        The following is equivalent to the above.

        >>> gidx = list(data.groupby('date').indices.values())
        ... apply_to_groups_reduce_jit(data['ret'], gidx, nanstd, None)
    """

    if config.disable_jit:
        return apply_to_groups(data, ginfo, function, *args, data2=data2)

    ginfo = _parse_group_info(data, ginfo)
    data, data2, args = _make_input_jit_compatible(data, data2, *args)

    if not n_ret:
        n_ret = 1 if data.ndim == 1 else data.shape[1]
    if n_ret == 1:
        retval = np.full((len(ginfo)), np.nan, dtype=data.dtype)
    else:
        retval = np.full((data.shape[0], n_ret), np.nan, dtype=data.dtype)

    if isinstance(ginfo, np.ndarray):  # group size
        if data2 is None:
            _apply_to_groups_reduce_gsize_jit_1(data, ginfo, function, retval, *args)
        else:
            _apply_to_groups_reduce_gsize_jit_2(data, data2, ginfo, function, retval, *args)
    elif isinstance(ginfo, list):  # group index list
        if data2 is None:
            _apply_to_groups_reduce_gidx_jit_1(data, List(ginfo), function, retval, *args)
        else:
            _apply_to_groups_reduce_gidx_jit_2(data, data2, List(ginfo), function, retval, *args)

    return retval.ravel() if n_ret == 1 else retval


################################################################################
#
# CLASSIFY/TRIM/WINSORIZE DATA
#
################################################################################

@njit(cache=True)
def _classify(array, by_array, quantiles, ascending=True):
    classes = np.full(len(array), np.nan)

    if np.all(np.isnan(by_array)):
        return classes

    values = np.nanquantile(by_array, quantiles)
    classes[array <= values[0]] = 0
    for i, v in enumerate(values):
        classes[array > v] = i + 1

    if not ascending:
        classes = len(quantiles) - classes

    return classes


def classify(array, split, ascending=True, ginfo=None, by_array=None):
    """Classify array.

    Classify (group) `array` into `split` classes based on its value.
    Class labels are set to 0, 1, ... where 0 corresponds to the lowest (highest) value group if
    `ascending` = True (False). If array contains nan, their classes are set to nan.

    Args:
        array: Nx1 ndarray or Series to be classified.
        split: Number of classes (for equal-size quantiles) or list of quantiles, e.g., (0.3, 0.7).
        ascending (bool): Sorting order.
        ginfo: Grouping information: integer (index level), str (column name), pandas GroupBy object, ndarray of
            group sizes, or list of group indexes. If given, `array` is classified within each group.
            See :func:`apply_to_groups` for more details.
        by_array: Array based on which cut points are determined. If None, `by_array` = `array`. E.g., array can be set
            to ME and by_array to NYSE-ME to group firms on size with NYSE-size cut points.

    Returns:
        Nx1 ndarray of classes.

    NOTE:
        If the array has one unique value, the class will be set to 0, and if the array has two unique values (binary
        variable), the class of the smaller value will be 0 and that of the larger value will be
        (number of quantiles - 1), when `ascending` = True.
        If the number of unique values is greater than 2 and smaller than the number of quantiles, the classes are not
        deterministic.

    Examples:
        >>> array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        ... array
        [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]

        Classify array into 5 equally-spaced groups.

        >>> classify(array, 5)
        [0., 0., 0., 1., 1., 2., 2., 3., 3., 4., 4.]

        Classify array into three groups that correspond to 0.3, 03-0.7, 0.7-1.0 quantiles.

        >>> classify(array, [0.3, 0.7])
        [0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2.]

        Classify array into three groups in descending order.

        >>> classify(array, [0.3, 0.7], ascending=False)
        [2., 2., 2., 2., 1., 1., 1., 1., 0., 0., 0.]

        >>> data = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
        ...      'me_nyse': [np.nan, 5000, 2000, 300, 150, np.nan, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... ).sort_index(level=['permno', 'date'])
        ... data
                             ret    me  me_nyse
        date       permno
        2023-03-31 10000   0.010   100      NaN
        2023-04-30 10000  -0.030   120      NaN
        2023-03-31 20000   0.050  5000 5000.000
        2023-04-30 20000  -0.010  4500 4500.000
        2023-03-31 30000   0.020  2000 2000.000
        2023-04-30 30000   0.030  2100 2100.000
        2023-03-31 40000   0.030   300  300.000
        2023-04-30 40000  -0.050   305  305.000
        2023-03-31 50000   0.110   150  150.000
        2023-04-30 50000   0.070   140  140.000

        Cross-sectionally classify the data into three groups ([0.3, 0.7, 1.0]) on 'me' using 'me_nyse' cut points.

        >>> data['me_cls'] = classify(data['me'], [0.3, 0.7], ginfo='date', by_array=data['me_nyse'])
        ... data
                             ret    me  me_nyse  me_cls
        date       permno
        2023-03-31 10000   0.010   100      NaN       0
        2023-04-30 10000  -0.030   120      NaN       0
        2023-03-31 20000   0.050  5000 5000.000       2
        2023-04-30 20000  -0.010  4500 4500.000       2
        2023-03-31 30000   0.020  2000 2000.000       1
        2023-04-30 30000   0.030  2100 2100.000       1
        2023-03-31 40000   0.030   300  300.000       1
        2023-04-30 40000  -0.050   305  305.000       1
        2023-03-31 50000   0.110   150  150.000       0
        2023-04-30 50000   0.070   140  140.000       0
    """

    if by_array is None:
        by_array = array

    # split to quantiles
    if is_int(split):  # number of classes
        quantiles = np.linspace(0, 1, split + 1)[1:-1]
    else:  # list or tuple of quantiles
        quantiles = np.array(split if split[-1] < 1 else split[:-1])

    if ginfo is None:
        array, by_array, _ = _make_input_jit_compatible(array, by_array)
        return _classify(array, by_array, quantiles, ascending)
    else:
        return apply_to_groups_jit(array, ginfo, _classify, 1, quantiles, ascending, data2=by_array)


@njit(cache=True)
def _trim(array, by_array, limits):
    retval = np.full(array.shape, True, dtype=np.bool_)

    if not np.isnan(limits[0]):
        lq = np.nanquantile(by_array, limits[0])
        retval &= array >= lq

    if not np.isnan(limits[1]):
        uq = np.nanquantile(by_array, 1 - limits[1])
        retval &= array <= uq

    return retval


def trim(array, limits, ginfo=None, by_array=None):
    """Trim array.

    Trim `array` values that are outside `limits`. The returned value is a bool array that indicates which to be
    trimmed (True to keep and False to remove).

    Args:
        array: Nx1 ndarray or Series to be trimmed.
        limits: A pair of quantiles, e.g., (0.1, 0.1) to trim top and bottom 10%. An element of `limits` can be set to
            None for one-sided trim.
        ginfo: Grouping information: integer (index level), str (column name), pandas GroupBy object, ndarray of
            group sizes, or list of group indexes. If given, `array` is trimmed within each group.
            See :func:`apply_to_groups` for more details.
        by_array: Array based on which cut points are determined. If None, `by_array` = `array`. E.g., `array` can be
            set to ME and `by_array` to NYSE-ME to remove small firms based on NYSE-size cut points.

    Returns:
        Nx1 bool ndarray. Elements corresponding to trimmed values are set to False.

    Examples:
        >>> array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        ... array
        [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]

        Trim top/bottom 10% values.

        >>> trim(array, [0.1, 0.1])
        [False,  True,  True,  True,  True,  True,  True,  True,  True, True, False]

        >>> data = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
        ...      'me_nyse': [np.nan, 5000, 2000, 300, 150, np.nan, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... ).sort_index(level=['permno', 'date'])
        ... data
                             ret    me  me_nyse
        date       permno
        2023-03-31 10000   0.010   100      NaN
        2023-04-30 10000  -0.030   120      NaN
        2023-03-31 20000   0.050  5000 5000.000
        2023-04-30 20000  -0.010  4500 4500.000
        2023-03-31 30000   0.020  2000 2000.000
        2023-04-30 30000   0.030  2100 2100.000
        2023-03-31 40000   0.030   300  300.000
        2023-04-30 40000  -0.050   305  305.000
        2023-03-31 50000   0.110   150  150.000
        2023-04-30 50000   0.070   140  140.000

        Trim smallest 20% cross-sectionally using 'me_nyse' cut points.

        >>> data['trimmed'] = trim(data['me'], [0.2, None], ginfo='date', by_array=data['me_nyse'])
        ... data
                             ret    me  me_nyse  trimmed
        date       permno
        2023-03-31 10000   0.010   100      NaN    False
        2023-04-30 10000  -0.030   120      NaN    False
        2023-03-31 20000   0.050  5000 5000.000     True
        2023-04-30 20000  -0.010  4500 4500.000     True
        2023-03-31 30000   0.020  2000 2000.000     True
        2023-04-30 30000   0.030  2100 2100.000     True
        2023-03-31 40000   0.030   300  300.000     True
        2023-04-30 40000  -0.050   305  305.000     True
        2023-03-31 50000   0.110   150  150.000    False
        2023-04-30 50000   0.070   140  140.000    False
    """

    if by_array is None:
        by_array = array

    limits = np.array(limits, dtype=float)

    if ginfo is None:
        array, by_array, _ = _make_input_jit_compatible(array, by_array)
        retval = _trim(array, by_array, limits)
    else:
        retval = apply_to_groups_jit(array, ginfo, _trim, 1, limits, data2=by_array)

    return retval.astype(bool)


def filter(data, on, limits, ginfo=None, by=None):
    """Filter data.

    Remove rows of `data`, where `data[on]` is outside `limits`.

    Args:
        data: DataFrame to be filtered.
        on: Column of `data` to filter `data` on.
        limits: A pair of quantiles, e.g., (0.1, 0.1) to remove top and bottom 10%. An element of `limits` can be set to
            None for one-sided filtering.
        ginfo: Grouping information: integer (index level), str (column name), pandas GroupBy object, ndarray of
            group sizes, or list of group indexes. If given, `data` is filtered within each group.
            See :func:`apply_to_groups` for more details.
        by: Column of `data` on which cut points are determined. If None, `by` = `on`. E.g., `on` can be set
            to 'me' and `by` to 'nyse_me' to remove small firms based on NYSE-size cut points.

    Returns:
        DataFrame. Filtered `data`.

    Examples:
        >>> data = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
        ...      'me_nyse': [np.nan, 5000, 2000, 300, 150, np.nan, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... ).sort_index(level=['permno', 'date'])
        ... data
                             ret    me  me_nyse
        date       permno
        2023-03-31 10000   0.010   100      NaN
        2023-04-30 10000  -0.030   120      NaN
        2023-03-31 20000   0.050  5000 5000.000
        2023-04-30 20000  -0.010  4500 4500.000
        2023-03-31 30000   0.020  2000 2000.000
        2023-04-30 30000   0.030  2100 2100.000
        2023-03-31 40000   0.030   300  300.000
        2023-04-30 40000  -0.050   305  305.000
        2023-03-31 50000   0.110   150  150.000
        2023-04-30 50000   0.070   140  140.000

        Remove smallest 20% using 'me_nyse' cut points.

        >>> filter(data, 'me', [0.2, None], ginfo='date', by='me_nyse')
                             ret    me  me_nyse
        date       permno
        2023-03-31 20000   0.050  5000 5000.000
        2023-04-30 20000  -0.010  4500 4500.000
        2023-03-31 30000   0.020  2000 2000.000
        2023-04-30 30000   0.030  2100 2100.000
        2023-03-31 40000   0.030   300  300.000
        2023-04-30 40000  -0.050   305  305.000
    """

    by = by or on
    trimmed = trim(data[on], limits, ginfo, data[by])
    return data[trimmed]


@njit(cache=True)
def _winsorize(array, by_array, limits):
    if not np.isnan(limits[0]):
        lq = np.nanquantile(by_array, limits[0])
        lq = np.min(array[array >= lq])
    else:
        lq = -np.inf

    if not np.isnan(limits[1]):
        uq = np.nanquantile(by_array, 1 - limits[1])
        uq = np.max(array[array <= uq])
    else:
        uq = np.inf

    return np.clip(array, lq, uq)


def winsorize(array, limits, ginfo=None, by_array=None):
    """Winsorize array.

    Winsorize `array` values that are outside `limits`.

    Args:
        array: Nx1 ndarray or Series to be winsorized.
        limits: A pair of quantiles, e.g., (0.1, 0.1) to winsorize top and bottom 10%. An element of `limits` can be
            set to None for one-sided winsorization.
        ginfo: Grouping information: integer (index level), str (column name), pandas GroupBy object, ndarray of
            group sizes, or list of group indexes. If given, `array` is winsorized within each group.
            See :func:`apply_to_groups` for more details.
        by_array: Array based on which cut points are determined. If None, `by_array` = `array`. E.g., `array` can be
            set to ME and `by_array` to NYSE-ME to winsorize large firms' weights based on NYSE-size cut points.

    Returns:
        Nx1 ndarray of winsorized values.

    Examples:
        >>> array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        ... array
        [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]

        Winsorize top/bottom 10% values.

        >>> winsorize(array, [0.1, 0.1])
        [ 2,  2,  3,  4,  5,  6,  7,  8,  9, 10, 10]

        >>> data = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
        ...      'me_nyse': [np.nan, 5000, 2000, 300, 150, np.nan, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... ).sort_index(level=['permno', 'date'])
        ... data
                             ret    me  me_nyse
        date       permno
        2023-03-31 10000   0.010   100      NaN
        2023-04-30 10000  -0.030   120      NaN
        2023-03-31 20000   0.050  5000 5000.000
        2023-04-30 20000  -0.010  4500 4500.000
        2023-03-31 30000   0.020  2000 2000.000
        2023-04-30 30000   0.030  2100 2100.000
        2023-03-31 40000   0.030   300  300.000
        2023-04-30 40000  -0.050   305  305.000
        2023-03-31 50000   0.110   150  150.000
        2023-04-30 50000   0.070   140  140.000

        Winsorize top 10% returns cross-sectionally.

        >>> data['ret_winsorized'] = winsorize(data['ret'], [None, 0.1], ginfo='date')
        ... data[['ret', 'ret_winsorized']].sort_index()
                             ret  ret_winsorized
        date       permno
        2023-03-31 10000   0.010           0.010
                   20000   0.050           0.050
                   30000   0.020           0.020
                   40000   0.030           0.030
                   50000   0.110           0.050
        2023-04-30 10000  -0.030          -0.030
                   20000  -0.010          -0.010
                   30000   0.030           0.030
                   40000  -0.050          -0.050
                   50000   0.070           0.030
    """

    if by_array is None:
        by_array = array

    limits = np.array(limits, dtype=float)

    if ginfo is None:
        array, by_array, _ = _make_input_jit_compatible(array, by_array)
        return _winsorize(array, by_array, limits)
    else:
        return apply_to_groups_jit(array, ginfo, _winsorize, 1, limits, data2=by_array)


################################################################################
#
# ANOMALY DETECTION
#
################################################################################

def _detect_anomalies_quantile(data, columns=None, types='change', lq=0.05, lmultiple=1, uq=0.95, umultiple=1,
                               min_neg=0.0005):
    """Detect anomalies using the quantile-based method.

    A data point `x_it` is deemed as an anomaly if it is outside the boundary values:

        * `lb = quantile(x, lq) - lmultiple * (median(x) - quantile(x, lq))`
        * `ub = quantile(x, uq) - umultiple * (median(x) - quantile(x, uq))`

    Args:
        data: DataFrame with index = date/id.
        columns: Columns to examine. If None, all numeric columns are examined.
        types: (List of) anomaly type(s). Possible values are 'change', 'value', and 'negative'. The 'change' detects
            anomalies in the percentage change of `x` and 'value' detects anomalies in `x` itself. The 'negative'
            treats negative values as anomalies if the variable has negative values with a frequency fewer than `min_neg`.
        lq: Quantile for the lower bound.
        lmultiple: Constant to be multiplied to (median - quantile) in the lower bound.
        uq: Quantile for the upper bound.
        umultiple: Constant to be multiplied to (median - quantile) in the upper bound.
        min_neg: Value to determine whether a variable should be non-negative. If number of negative values / number of
            samples < `min_neg`, the variable is considered non-negative and the negative values are treated as anomalies.

    Returns:
        Numpy 2-D bool array of anomaly indicators.
    """

    types = to_list(types)
    columns = columns or data.columns

    anomaly = np.full((data.shape[0], len(columns)), False)

    if 'change' in types:
        # t-1 to t change
        data1 = data[columns].groupby(data.index.names[-1]).pct_change()
        data1.replace([np.inf, -np.inf], np.nan, inplace=True)

        for i, col in enumerate(columns):
            m = data1[col].median()

            q = data1[col].quantile(lq)
            anomaly_i = data1[col] < q - lmultiple * (m - q)

            q = data1[col].quantile(uq)
            anomaly_i |= data1[col] > q - umultiple * (m - q)

            anomaly[:, i] |= anomaly_i

        # t to t+1 change
        data1 = data[columns].groupby(data.index.names[-1]).pct_change(-1)
        data1.replace([np.inf, -np.inf], np.nan, inplace=True)
        for i, col in enumerate(columns):
            m = data1[col].median()

            q = data1[col].quantile(lq)
            anomaly_i = data1[col] < q - lmultiple * (m - q)

            q = data1[col].quantile(uq)
            anomaly_i |= data1[col] > q - umultiple * (m - q)

            anomaly[:, i] &= anomaly_i

    if 'value' in types:
        for i, col in enumerate(columns):
            m = data[col].median()

            q = data[col].quantile(lq)
            anomaly_i = data[col] < q - lmultiple * (m - q)

            q = data[col].quantile(uq)
            anomaly_i |= data[col] > q - umultiple * (m - q)

            anomaly[:, i] |= anomaly_i

    if 'negative' in types:
        negative_cols = []
        for i, col in enumerate(columns):
            negative = data[col] < 0
            if negative.any() and (negative.sum() / data[col].count() < min_neg):
                log(f'{col}: negative values will be treated as anomalies.')
                anomaly[:, i] |= negative
                negative_cols.append(col)

        log(list(np.sort(negative_cols)))
    return anomaly


def _detect_anomalies_ml(data, columns=None, algorithm='IsolationForest', **kwargs):
    """Detect anomalies using a machine learning method.

    Anomalies are detected using one of three algorithms, 'IsolationForest', 'RobustCovariance', and 'LocalOutlierFactor'.
    For details of each method, refer to the scikit-learn documentation.
    This function detects anomalies in the changes of the variable, (x_it - x_it-1) / x_it-1. Missing values in a column
    are replaced by the median.

    References:

        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest.fit_predict

        https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope

        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor

    Args:
        data: DataFrame with index = date/id.
        columns: Columns to examine. If None, all numeric columns are examined.
        algorithm: 'IsolationForest', 'RobustCovariance', or 'LocalOutlierFactor'.
        kwargs: Dict of arguments for the given algorithm. See the scikit-learn documentation for the arguments.

    Returns:
        Numpy 2-D bool array of anomaly indicators.
    """

    columns = columns or data.columns
    X = data[columns]

    # Calculate change per id.
    X = X.groupby(X.index.names[-1]).pct_change()

    # Replace inf and nan with median.
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X:
        X[col] = X[col].fillna(X[col].median())
    X = X.dropna(axis=1)  # There can still be nan if all values of a column are nan.

    # Anomaly detection
    if algorithm == 'IsolationForest':
        model = IsolationForest(**kwargs)
        y_pred = model.fit(X).predict(X)
    elif algorithm == 'RobustCovariance':
        model = EllipticEnvelope(**kwargs)
        y_pred = model.fit(X).predict(X)
    elif algorithm == 'LocalOutlierFactor':
        model = LocalOutlierFactor(**kwargs)
        y_pred = model.fit_predict(X)
    else:
        raise ValueError(f'Unsupported anomaly detection method: {algorithm}')

    anomaly = ~data.isna() & (y_pred == -1).reshape(-1, 1)
    return anomaly


def _detect_anomalies(data, columns=None, method='Quantile', **kwargs):
    if method == 'Quantile':
        return _detect_anomalies_quantile(data, columns, **kwargs)
    elif method in ('IsolationForest', 'RobustCovariance', 'LocalOutlierFactor'):
        return _detect_anomalies_ml(data, columns, method, **kwargs)
    else:
        raise ValueError(f'Undefined anomaly detection method: {method}.')


################################################################################
#
# MERGE/POPULATE
#
################################################################################

def merge(left, right, on=None, right_on=None, how='left', drop_duplicates='right', suffixes=None, method=None):
    """Merge two data sets.

    This is similar to ``pd.merge()``, but often much faster and less memory-hungry when merging left.
    Also, the index of `left` is always retained.

    Args:
        left: Series or DataFrame. Left data to merge.
        right: Series or DataFrame, Right data to merge.
        on: (List of) column(s) to merge on. If None, merge on index.
        right_on: (List of) column(s) of `right` to merge on. If None, `right_on` = `on`.
        how: Merge method: 'inner', 'outer', 'left', or 'right'.
        drop_duplicates: how to handle duplicate columns. 'left': keep right, 'right': keep left,
            None: keep both. If None, `suffixes` should be provided.
        suffixes: A tuple of suffixes for duplicate columns, e.g., suffixes=('_x', '_y') will add '_x' and '_y'
            to the left and right duplicate columns, respectively.
        method: None or 'pandas'. None uses an internal merge algorithm for left-merge; 'pandas' uses ``pd.merge()``
            internally. If `how` is not 'left', this option is ignored and ``pd.merge()`` is always used.

    Returns:
        Merged DataFrame.

    Note:
        The internal algorithm is much faster and more memory-efficient than ``pd.merge()`` especially when
        `how` = 'left' and `right` data does not have many columns. In other cases, it could be slower. Try both
        `method` = None and 'merge', and choose the faster method.

    Warnings:
        The `left` and `right` could be modified internally.

    Examples:
        >>> data1 = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... )
        ... data1
                             ret    me
        date       permno
        2023-03-31 10000   0.010   100
                   20000   0.050  5000
                   30000   0.020  2000
                   40000   0.030   300
                   50000   0.110   150
        2023-04-30 10000  -0.030   120
                   20000  -0.010  4500
                   30000   0.030  2100
                   40000  -0.050   305
                   50000   0.070   140

        >>> data2 = pd.DataFrame(
        ...     {'me': [120, 4500, 2100, 305, 140],
        ...      'me_nyse': [np.nan, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... )
        ... data2
                             me  me_nyse
        date       permno
        2023-04-30 10000    120      NaN
                   20000   4500 4500.000
                   30000   2100 2100.000
                   40000    305  305.000
                   50000    140  140.000

        >>> merge(data1, data2)
                             ret    me  me_nyse
        date       permno
        2023-03-31 10000   0.010   100      NaN
                   20000   0.050  5000      NaN
                   30000   0.020  2000      NaN
                   40000   0.030   300      NaN
                   50000   0.110   150      NaN
        2023-04-30 10000  -0.030   120      NaN
                   20000  -0.010  4500 4500.000
                   30000   0.030  2100 2100.000
                   40000  -0.050   305  305.000
                   50000   0.070   140  140.000

        >>> merge(data1, data2, how='inner')
                             ret    me  me_nyse
        date       permno
        2023-04-30 10000  -0.030   120      NaN
                   20000  -0.010  4500 4500.000
                   30000   0.030  2100 2100.000
                   40000  -0.050   305  305.000
                   50000   0.070   140  140.000

        >>> merge(data1, data2, drop_duplicates='left')
                             ret       me  me_nyse
        date       permno
        2023-03-31 10000   0.010      NaN      NaN
                   20000   0.050      NaN      NaN
                   30000   0.020      NaN      NaN
                   40000   0.030      NaN      NaN
                   50000   0.110      NaN      NaN
        2023-04-30 10000  -0.030  120.000      NaN
                   20000  -0.010 4500.000 4500.000
                   30000   0.030 2100.000 2100.000
                   40000  -0.050  305.000  305.000
                   50000   0.070  140.000  140.000
    """

    debug('merge start')

    def _set_index(data, on):
        index_names = data.index.names

        index_to_keep = [i for i in index_names if i in on]
        if index_to_keep == on:
            is_index = True
            index_to_reset = list(set(index_names).difference(set(on)))
            if index_to_reset:
                data.reset_index(level=index_to_reset, inplace=True)
        else:
            is_index = False
            if index_names[0]:
                index_to_reset = index_names
                data.reset_index(inplace=True)
            else:
                index_to_reset = []
            data.set_index(on, inplace=True)

        return data, is_index, index_to_reset
        # if on == index_names:
        #     is_index = True
        # else:
        #     is_index = True
        #     for i in on:
        #         is_index &= i in index_names

        # index_to_reset = [i for i in index_names if i not in on]
        # print(is_index)
        # print(index_to_reset)
        # print(index_names)
        # print(on)

    index_names = left.index.names

    @njit(cache=True)
    def _merge_numeric(true_index, rcol, rcol2):
        for i in range(true_index.shape[0]):
            if true_index[i] >= 0:
                rcol2[i] = rcol[int(true_index[i])]
        return rcol2

    if on is None:
        on = index_names
    elif isinstance(on, str):
        on = [on]

    right_on = right_on or on
    if isinstance(right_on, str):
        right_on = [right_on]
    if isinstance(right, pd.Series):
        right = right.to_frame()

    columns_to_merge = [col for col in right.columns if col not in right_on]
    dup_columns = [col for col in columns_to_merge if col in left.columns]

    left, is_index, index_to_reset = _set_index(left, on)
    right, _, _ = _set_index(right, right_on)

    if dup_columns:
        if drop_duplicates == 'right':
            columns_to_merge = [col for col in columns_to_merge if col not in dup_columns]
        elif drop_duplicates == 'left':
            drop_columns(left, dup_columns)
        else:
            suffixes = suffixes or ('_x', '_y')
            left_dup_columns = {col: col + suffixes[0] for col in dup_columns}
            left.rename(columns=left_dup_columns, inplace=True)
            right_dup_columns = {col: col + suffixes[1] for col in dup_columns}
            right.rename(columns=right_dup_columns, inplace=True)
            columns_to_merge = [right_dup_columns[col] if col in dup_columns else col for col in columns_to_merge]

    debug('merge core start')
    if how == 'left':
        if method == 'pandas':
            left1 = left.iloc[:, [0]].merge(right.loc[:, columns_to_merge], left_on=on, right_on=right_on, how='left')
            for col in columns_to_merge:
                left[col] = left1[col].to_numpy()
        else:
            if len(columns_to_merge) == 1:
                left[columns_to_merge] = right[columns_to_merge]
            else:
                left['true'] = pd.Series(np.arange(right.shape[0]), index=right.index)
                true_index = left['true'].to_numpy()
                for col in columns_to_merge:
                    if is_numeric(right[col]) or (right[col].dtype in ('<M8[ns]', bool)):
                        dtype = config.float_type if is_int(right[col]) else right[col].dtype
                        rcol2 = np.full(true_index.shape[0], None, dtype=dtype)
                        left[col] = _merge_numeric(true_index, right[col].to_numpy(), rcol2)
                    else:
                        left[col] = right[col]
                        # rcol = right[col].to_numpy().astype(np.unicode_)
                        # rcol2 = np.full(true_index.shape[0], 'None', dtype=rcol.dtype)
                        # left[col] = _merge_object(true_index, rcol, rcol2)
                        # if right[col].dtype == 'boolean':
                        #     left[col] = np.where(left[col]=='True', True, np.where(left[col]=='False', False, left[col]))
                        # left[col] = left[col].replace(('None', '<NA>'), None).astype(right[col].dtype)
                del left['true']
    else:
        left = left.merge(right[columns_to_merge], left_on=on, right_on=right_on, how=how)

    debug('merge core end')

    if index_to_reset and index_to_reset[0]:  # index has a name.
        left.reset_index(inplace=True)
        left.set_index(index_names, inplace=True)
    elif not is_index:
        left.reset_index(inplace=True)

    debug('merge end')
    return left


def populate(data, freq, method='ffill', limit=None, new_date_idx=None):
    """Populate data.

    Populate `data` to `freq` frequency.

    Args:
        data: DataFrame with index = date/id, sorted on id/date.
        freq: Frequency to populate: ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        method: Filling method for newly added rows. 'ffill': forward fill, None: nan.
        limit: Maximum number of rows to forward-fill. Default to None (no fill).
        new_date_idx: Name of the new (populated) date index. If None, use the current date index name. If given,
            the original date index is kept as a column.

    Returns:
        Populated data with index = new_date/id.

    Examples:
        >>> data = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03],
        ...      'me': [100, 5000, 120, 4500],
        ...      },
        ...     index=pd.MultiIndex.from_product([pd.to_datetime(['2023-03-31', '2024-03-31']), [10000, 20000]],
        ...                                      names=['date', 'permno'])
        ... )
        ... data
                            ret    me
        date       permno
        2023-03-31 10000  0.010   100
                   20000  0.050  5000
        2024-03-31 10000  0.020   120
                   20000  0.030  4500

        Populate to monthly and forward-fill up to 12 months.

        >>> populate(data, MONTHLY, limit=12)
                            ret       me
        date       permno
        2023-03-31 10000  0.010  100.000
        2023-04-30 10000  0.010  100.000
        2023-05-31 10000  0.010  100.000
        2023-06-30 10000  0.010  100.000
        2023-07-31 10000  0.010  100.000
        2023-08-31 10000  0.010  100.000
        2023-09-30 10000  0.010  100.000
        2023-10-31 10000  0.010  100.000
        2023-11-30 10000  0.010  100.000
        2023-12-31 10000  0.010  100.000
        2024-01-31 10000  0.010  100.000
        2024-02-29 10000  0.010  100.000
        2024-03-31 10000  0.020  120.000
        2023-03-31 20000  0.050 5000.000
        2023-04-30 20000  0.050 5000.000
        2023-05-31 20000  0.050 5000.000
        2023-06-30 20000  0.050 5000.000
        2023-07-31 20000  0.050 5000.000
        2023-08-31 20000  0.050 5000.000
        2023-09-30 20000  0.050 5000.000
        2023-10-31 20000  0.050 5000.000
        2023-11-30 20000  0.050 5000.000
        2023-12-31 20000  0.050 5000.000
        2024-01-31 20000  0.050 5000.000
        2024-02-29 20000  0.050 5000.000
        2024-03-31 20000  0.030 4500.000
    """

    @njit(cache=True)
    def _populate_inner(ids, dates, dates_adj, gsize, all_dates, limit):
        def inner(id, dates, dates_adj, all_dates, limit):
            new_dates = all_dates[(all_dates >= dates_adj[0]) & (all_dates <= dates_adj[-1])]
            org_dates = np.full_like(new_dates, NaT)
            ids = np.full(new_dates.shape, id)

            if new_dates.shape[0] == 1:
                return ids, new_dates, new_dates

            j, k = 0, 0
            for i in prange(new_dates.shape[0]):
                if (new_dates[i] >= dates_adj[j]) and (new_dates[i] < dates_adj[j + 1]):
                    if k <= limit:
                        org_dates[i] = dates[j]
                    k += 1
                else:
                    j += 1
                    k = 1
                    org_dates[i] = dates[j]

            return ids, new_dates, org_dates

        idx = np.full(1 + gsize.shape[0], 0)
        idx[1:] = np.cumsum(gsize)
        retval = []
        for i in prange(gsize.shape[0]):
            retval.append(inner(ids[i], dates[idx[i]:idx[i + 1]], dates_adj[idx[i]:idx[i + 1]], all_dates, limit))
        return retval

    if (not method) or (not limit):
        limit = 0

    id_index = data.index.get_level_values(1)
    if (not id_index.is_monotonic_increasing) and (not id_index.is_monotonic_decreasing):
        raise ValueError('populate: The input data must be sorted on ID.')

    date_idx, id_idx = data.index.names
    if (not new_date_idx) or (new_date_idx == date_idx):
        new_date_idx = date_idx + '_xxx'

    # Resample all dates.
    data_ = data.index.to_frame(False)
    if freq == MONTHLY:
        data_['date_adj'] = to_month_end(data_[date_idx])
    else:
        data_['date_adj'] = data_[date_idx]

    NaT = np.datetime64('NaT').astype(data_[date_idx].dtype)

    tmp = pd.Series(index=data_['date_adj'].unique(), dtype=int).resample(freq_map[freq]).asfreq()
    all_dates = np.sort(tmp.index.to_numpy())

    gb = data_.groupby(id_idx)
    gsize = gb.size().to_numpy()
    ids = np.array(list(gb.indices.keys()))
    retval = _populate_inner(ids, data_[date_idx].to_numpy(), data_['date_adj'].to_numpy(), gsize, all_dates, limit)
    ids, new_dates, org_dates = _parse_retval(retval)

    new_data = pd.DataFrame({new_date_idx: new_dates, id_idx: ids, date_idx: org_dates})
    new_data[id_idx] = new_data[id_idx].astype(
        data.index.dtypes.iloc[1])  # string can be changed to object during resample.

    # new_data = new_data.merge(data, left_on=[date_idx, id_idx], right_on=[date_idx, id_idx], how='left')
    new_data = merge(new_data, data, on=[date_idx, id_idx], how='left')

    if new_date_idx == date_idx + '_xxx':
        del new_data[date_idx]
        new_data.rename(columns={new_date_idx: date_idx}, inplace=True)
        new_data.set_index([date_idx, id_idx], inplace=True)
    else:
        new_data.set_index([new_date_idx, id_idx], inplace=True)

    return new_data


def shift(data, n, cols=None, excl_cols=None):
    """Shift data.

    Shift `data` by `n`. If `cols` is given, only `cols` columns are shifted. If `excl_cols` is given, columns
    excluding `excl_cols` are shifted. Either `cols` or `excl_cols` should be None. The shifted data contains both
    shifted and not-shifted columns. If `data` has a MultiIndex of date/id, data is shifted within each id.

    Args:
        data: Series or DataFrame with index = date or date/id. If index = date/id, `data` must be sorted on id/date.
        n: Shift size
        cols: Columns to shift.
        excl_cols: Columns to not shift.

    Returns:
        Series or DataFrame. Shifted `data`.
    """

    assert (cols is None) or (excl_cols is None)

    if isinstance(data.index, pd.MultiIndex):
        shift_data = data.groupby(level=1).shift(n)
    else:
        shift_data = data.shift(n)

    if cols:
        excl_cols = list(data.columns.difference(cols))

    if excl_cols:
        shift_data[excl_cols] = data[excl_cols]

    # shift_data = data.shift(n)
    # if cols:
    #     excl_cols = list(data.columns.difference(cols))
    #
    # if excl_cols:
    #     shift_data[excl_cols] = data[excl_cols]
    #
    # if isinstance(data.index, pd.MultiIndex):
    #     ids = data.index.get_level_values(1)
    #     shift_data.loc[ids != np.roll(ids, n)] = np.nan

    return shift_data


################################################################################
#
# DATA INSPECTION/COMPARISON
#
################################################################################

def inspect_data(data, option=['summary'], date_col=None, id_col=None):
    """Inspect data.

    This function inspects a panel data, `data`, and print the results.

    Args:
        data: DataFrame. It should have date and id columns or index = date/id.
        option: List of items to display. See below for available options.
        date_col: Date column. If None, `date.index[0]` is assumed to be date.
        id_col: ID column. If None, `date.index[1]` is assumed to be id.

    .. csv-table::
        :header: Option, Items displayed

        'summary', "Data shape, number of unique dates, and number of unique ids."
        'id_count`, Number of ids per date.
        'nans', Number of nans and infs per column.
        'stats', Descriptive statistics. Same as ``data.describe()``.
    """

    max_rows = pd.get_option('display.max_rows')  # get current setting
    pd.set_option('display.max_rows', data.shape[1])  # set to num. of columns.

    if 'summary' in option:
        if date_col:
            dates = data[date_col]
        else:
            date_col = data.index.names[0]
            dates = data.index.get_level_values(0)

        if id_col:
            ids = data[id_col]
        else:
            ids = data.index.get_level_values(1)

        log(f'Shape: {data.shape}')
        log(f'No. unique dates: {len(dates.unique())}, Date range: {dates.min()} - {dates.max()}')
        log(f'No. unique ids: {len(ids.unique())}')

    if 'id_count' in option:
        log(f'No. ids per date:\n {data.groupby(date_col).size()}')

    if 'nans' in option:
        log('\n', header=False)
        log('Nans, Infs')
        nans = data.isna().sum()
        nans = pd.concat([nans, nans / data.shape[0]], axis=1, keys=['no. nans', 'percentage'])

        infs = ((data == np.inf) | (data == -np.inf)).sum()
        infs = pd.concat([infs, infs / data.shape[0]], axis=1, keys=['no. infs', 'percentage'])

        log(pd.concat([nans, infs], axis=1), header=False)

    if 'stats' in option:
        log('\n', header=False)
        log('Descriptive statistics')
        desc_list = []
        for col in data:
            desc_list.append(data[col].describe([0.001, 0.01, .25, .5, .75, 0.99, 0.999]))
        log(pd.concat(desc_list, axis=1).T, header=False)
        # log(data.describe([0.001, 0.01, .25, .5, .75, 0.99, 0.999]).T, header=False)  # Slow and memory hungrier

    pd.set_option('display.max_rows', max_rows)  # back to previous setting


def compare_data(data1, data2=None, on=None, how='inner', tolerance=0.01, suffixes=('_x', '_y'), returns=False):
    """Compare two data sets.

    This function compares the common columns of `data1` and `data2`.
    This is similar to ``data1.compare(data2)``, but `data1` and `data2` are not required to have the same index and
    columns: Data sets are first merged and only common columns are compared. Also, a tolerance can be set to determine
    whether two values are identical. Whether two columns are identical within the tolerance ('match'), their
    correlation ('corr'), and the number of nans in `data1` and `data2` ('nan_x', 'nan_y') are printed.

    Args:
        data1: DataFrame for comparison.
        data2: DataFrame for comparison. If None, `data1` is assumed to be a merged dataset of `data1` and `data2`.
            If `data1` is a merged dataset, `on` and `how` have no effect.
        on: A column or a list of columns to merge data sets on. If None, data sets will be merged on index.
        how: How to merge: 'inner', 'outer', 'left', or 'right'. If 'inner', only common indexes are compared.
        tolerance: Tolerance level to determine equality. Two values, `val1` and `val2`, are considered to be identical
            if `abs((val1 - val2) / val2) < tolerance`.
        suffixes: suffixes to add to common columns or suffixes used in the merged dataset.
            `suffixes[0]`: suffix for `data1`, `suffixes[1]`: suffix for `data2`.
        returns: If True, return the comparison results and merged data.

    Returns:
        * Comparison result. DataFrame with index = compared columns and columns = ['match', 'corr', 'nan_x', 'nan_y'].
        * Merged DataFrame of `data1` and `data2`.

    Examples:
        >>> data1 = pd.DataFrame(
        ...     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... )
        ... data2 = pd.DataFrame(
        ...     {'ret': [0.00, np.nan, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
        ...      },
        ...     index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
        ...                                      names=['date', 'permno'])
        ... )
        ... compare_data(data1, data2)
        column                matched     corr    nan_x    nan_y
        ret                   0.88889  0.99773       0       1
    """

    if data2 is None:  # data1 is already a merged dataset.
        data = data1
    else:
        on = on or data1.index.names  # if None, merge on index.
        data = merge(data1.copy(), data2.copy(), on=on, how=how, drop_duplicates=None, suffixes=suffixes)
        data.sort_index(axis=1, inplace=True)

    item_list, match_list, corr_list, nan_x_list, nan_y_list = [], [], [], [], []
    log(f"{'column': <20}  {'matched': >7}  {'corr': >7}  {'nan_x': >7}  {'nan_y': >7}", header=False)
    for col in data.columns:
        if (col[-2:] == suffixes[0]):
            if not is_numeric(data[col]):
                continue
            col2 = col[:-2] + suffixes[1]
            if col2 not in data:
                continue
            pair = data[[col, col2]]
            nan_x = pair[col].isna().sum()
            nan_y = pair[col2].isna().sum()
            pair = pair[~pair.isin([np.nan, np.inf, -np.inf]).any(axis=1)]  # drop nan or inf
            corr = pair[col].corr(pair[col2])  # np.corrcoef(pair.T)[0, 1]
            pair['diff'] = (pair[col] - pair[col2]).abs()
            matchidx = ((pair[col2].abs() < 1e-5) & (pair['diff'] < 1e-5)) | (
                        pair['diff'] / pair[col2].abs() < tolerance)
            match = matchidx.sum() / len(pair)

            item_list.append(col[:-2])
            match_list.append(match)
            corr_list.append(corr)
            nan_x_list.append(nan_x)
            nan_y_list.append(nan_y)
            log(f'{col[:-2]: <20} {match: 7.5f} {corr: 7.5f} {nan_x: 7.0f} {nan_y: 7.0f}', header=False)

    if returns:
        compare_result = pd.DataFrame(
            {'match': match_list, 'corr': corr_list, 'nan_x': nan_x_list, 'nan_y': nan_y_list},
            index=pd.Index(item_list, name='column'))
        return compare_result, data


################################################################################
#
# AUX FUNCTIONS
#
################################################################################

def to_month_end(date):
    """Shift dates to the last dates of the same month.

    Args:
        date: Datetime Series.

    Returns:
        Datetime Series shifted to month end.
    """

    return date + MonthEnd(0)


def add_months(date, months, to_month_end=True):
    """Add months to dates.

    Args:
        date: Datetime Series.
        months: Months to add. Can be negative.
        to_month_end: If True, returned dates are end-of-month dates.

    Returns:
        Datetime Series of (`date` + `months`). Dates are end-of-month dates if `to_month_end` = True.
    """

    if to_month_end:
        return date + MonthEnd(months)
    else:
        return date + pd.DateOffset(months=months)


################################################################################
#
# DEPRECATED FUNCTIONS
#
################################################################################

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def _populate(data, id_idx, freq, method, limit):
    gb = data.groupby(id_idx)
    retval = []
    for k, g in gb:
        if method == 'ffill':
            retval.append(g.resample(freq_map[freq]).ffill(limit))
        else:  # fill with None
            retval.append(g.resample(freq_map[freq]).asfreq())
        retval[-1][id_idx] = k

    return pd.concat(retval)


def _populate_deprecated(data, freq, method='ffill', limit=None, new_date_idx=None):
    """[REPLACED BY A NEW FASTER FUNCTION] Populate data.

    Populate `data` to `freq` frequency.

    Args:
        data: DataFrame with index = date/id.
        freq: Frequency to populate: ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        method: Filling method for newly added rows. 'ffill': forward fill, None: nan.
        limit: Maximum number of rows to forward-fill.
        new_date_idx: Name of the new (populated) date index. If None, use the current date index name. If given,
            the original date index is kept as a column.

    Returns:
        Populated data with index = new_date/id.

    References:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html
    """

    date_idx, id_idx = data.index.names
    new_date_idx = new_date_idx or date_idx
    if date_idx == new_date_idx:
        new_date_idx = new_date_idx + '_x'

    pop_data = data.index.to_frame()
    pop_data[new_date_idx] = pop_data[date_idx]
    pop_data.set_index(new_date_idx, inplace=True)

    # Split ids into n_cpu groups.
    n_cpu = cpu_count()
    id_groups = np.array_split(pop_data[id_idx].unique(), n_cpu)

    # Create a process for each id group.
    retval = []
    futures = []
    with ProcessPoolExecutor() as executor:
        for ids in id_groups:
            data_ = pop_data[pop_data[id_idx].isin(ids)]
            futures.append(executor.submit(_populate, data_, id_idx, freq, method, limit))

        for f in as_completed(futures):
            retval.append(f.result())

    pop_data = pd.concat(retval).reset_index()
    pop_data[id_idx] = pop_data[id_idx].astype(data.index.dtypes.iloc[1])
    data = pop_data.merge(data, on=[date_idx, id_idx], how='left')
    if new_date_idx == date_idx + '_x':
        del data[date_idx]
        data.rename(columns={new_date_idx: date_idx}, inplace=True)
        new_date_idx = date_idx
    data.set_index([new_date_idx, id_idx], inplace=True)
    data.sort_index(level=[id_idx, new_date_idx], inplace=True)

    return data


if __name__ == '__main__':
    os.chdir('../')

    # data = pd.DataFrame(
    #     {'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, -0.01, 0.03, -0.05, 0.07],
    #      'me': [100, 5000, 2000, 300, 150, 120, 4500, 2100, 305, 140],
    #      'me_nyse': [np.nan, 5000, 2000, 300, 150, np.nan, 4500, 2100, 305, 140],
    #      },
    #      index=pd.MultiIndex.from_product([['2023-03-31', '2023-04-30'], [10000, 20000, 30000, 40000, 50000]],
    #                                         names=['date', 'permno'])
    # ).sort_index(level=['permno', 'date'])
