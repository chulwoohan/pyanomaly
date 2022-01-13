"""This module defines various toolkits for data-handling.

Rolling Application of Function
    The functions below apply a function to each group of a grouped data. For the differences,
    see the documentation of each function.

    * ``groupby_apply()``
    * ``groupby_apply_np()``
    * ``rolling_apply()``
    * ``rolling_apply_np()``

Grouping/Filtering/Trimming/Winsorization
    * ``classify()``: Classify data cross-sectionally.
    * ``filter_n()``: Filter data on a column.
    * ``filter()``: Filter data on a column with cut points based on another column.
    * ``trim()``: Trim data.
    * ``winsorize()``: Winsorize data.

Data Inspection/Comparison
    * ``inspect_data()``: Inspect a dataset.
    * ``compare_data()``: Compare two data sets.

Auxiliary Functions
    * ``populate()``: Populate data.
    * ``to_month_end()``: Shift dates to the last dates of the same month.
    * ``add_months()``: Add months to dates.
"""


import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import scipy.stats.mstats as mstats

from pyanomaly.globals import *


################################################################################
#
# ROLLING APPLICATION OF FUNCTION
#
################################################################################
def _parse_retval(retval):
    """Parse returns from groupby_apply or groupby_apply_np.

    Args:
        retval: List of one of the followings:
            i) (tuple of) float
            ii) (tuple of) ndarray
            iii) (tuple of) dataframe/series.

    Returns:
        Concatenated value of `retval`. The number of returns is equal to the length of `retval[0]`.
    """

    if isinstance(retval[0], tuple):  # multiple returns
        n_ret = len(retval[0])
        retvals = [[] for _ in range(n_ret)]
        for retval_ in retval:
            for i in range(n_ret):
                retvals[i].append(retval_[i])

        if isinstance(retvals[0][0], float):
            for i in range(n_ret):
                retvals[i] = np.array(retvals[i]).reshape((-1,1))
        elif isinstance(retvals[0][0], np.ndarray):
            for i in range(n_ret):
                retvals[i] = np.concatenate(retvals[i])
        else:
            for i in range(n_ret):
                retvals[i] = pd.concat(retvals[i])

        return tuple(retvals)
    elif isinstance(retval[0], float):
        return np.array(retval).reshape((-1, 1))
    elif isinstance(retval[0], np.ndarray):
        return np.concatenate(retval)
    else:
        return pd.concat(retval)

def groupby_apply(gb, fcn, *args):
    """Apply `fcn` to groups in `gb`.

    Args:
        gb: pd.GroupBy object.
        fcn: Function to apply to groups. Function arguments should be `(gbdata, \*args)`,
            where `gbdata` is an element of `gb`.

    Returns:
        Concatenated outputs of `fcn`.
    """

    retval = [fcn(v, *args) for k, v in gb]
    return _parse_retval(retval)


def groupby_apply_np(gb, fcn, *args):
    """Apply `fcn` to the values of the groups in `gb`.

    This is the same as ``groupby_apply()`` except that the first argument of `fcn` is the values of the grouped data
    (ndarray). Use this function if `fcn` cannot receive DataFrame as an argument, e.g., when `fcn` is wrapped by
    ``numpy.jit``.

    NOTE: The groupby operation must not change the order of the data as the input to `fcn`, `data.values`, does not
        have an index. E.g., if `data` is sorted on `id`/`date`, grouping `data` by `id` is fine but not by `date`.

    Args:
        gb: pd.GroupBy object.
        fcn: Function to apply to groups. Function arguments should be `(gbdata_values, \*args)`,
            where `gbdata_values` is the values of an element of `gb`.

    Returns:
        Concatenated outputs of `fcn`.
    """

    retval = [fcn(v.values, *args) for k, v in gb]
    return _parse_retval(retval)


def rolling_apply(data, by=None, fcn=None, *args):
    """Apply `fcn` to each group of data grouped by `by`.

    This is similar to ``groupby_apply()``: ``groupby_apply()`` receives pd.GroupBy object as input whereas this
    function performs ``pd.groupby()`` inside.

    Args:
        data: DataFrame to be grouped.
        by: Column by which `data` is grouped. If None, `data` is grouped by the last index.
        fcn: Function to apply to groups.  Function arguments should be `(gbdata, \*args)`,
            where `gbdata` is an element of the grouped data.

    Returns:
        Concatenated outputs of `fcn`.
    """

    by = by or data.index.names[-1]
    return groupby_apply(data.groupby(by), fcn, *args)


def rolling_apply_np(data, by=None, fcn=None, *args):
    """Apply `fcn` to each group of data grouped by `by`.

    This is the same as ``rolling_apply()`` except that the first argument of `fcn` is the values of the grouped data
    (ndarray). Use this function if `fcn` cannot receive DataFrame as an argument, e.g., when `fcn` is wrapped by
    ``numpy.jit``.

    NOTE: The groupby operation must not change the order of the data as the input to `fcn`, `data.values`, does not
        have an index. E.g., if `data` is sorted on `id`/`date`, grouping `data` by `id` is fine but not by `date`.

    Args:
        data: DataFrame to be grouped.
        by: Column by which `data` is grouped. If None, `data` is grouped by the last index, i.e., groupby(level=-1).
        fcn: Function to apply to groups. Function arguments should be `(gbdata_values, \*args)`,
            where `gbdata_values` is the values of an element of the grouped data.


    Returns:
        Concatenated outputs of `fcn`.
    """

    by = by or data.index.names[-1]
    return groupby_apply_np(data.groupby(by), fcn, *args)


# @staticmethod
# # @multiprocessing
# def rolling_apply1(cd, fcn, n_retval):
#     """Apply fcn in every month. This function runs a loop over permno and can be used when the calculation inside
#     fcn requires data over several months.
#
#     Args:
#         cd: crspd.data
#         data_columns: columns of cd that are used as input to fcn.
#         fcn(data): function to apply to a particular permno.
#             data: crspd.data[columns] for a given permno, where columns indicate the data used in fcn.
#             returns: num_months x num_returns ndarray.
#         n_retval: number of returns from fcn.
#
#     Returns:
#         (permno/mon) x num_returns ndarray.
#     """
#     ngroups = cd.groupby(['permno', 'mon']).ngroups
#     gsize = cd.groupby(['permno']).size().values
#     data = cd.values
#     mons = cd['mon'].values
#     # isnan = np.isnan(data).any(axis=1)
#
#     @njit
#     def rolling_apply1_(data, gsize, ngroups, mons):
#         retval = np.full((ngroups, n_retval), np.nan)
#
#         j, idx0 = 0, 0
#         for i in range(gsize.shape[0]):
#             data_ = data[idx0:idx0 + gsize[i]]
#             n_mon = np.unique(mons[idx0:idx0 + gsize[i]]).shape[0]
#
#             retval_ = fcn(data_)
#             if n_retval == 1:
#                 retval[j:j + n_mon, 0] = retval_
#             else:
#                 for k in range(n_retval):
#                     retval[j:j + n_mon, k] = retval_[k]
#
#             idx0 += gsize[i]
#             j += n_mon
#
#         return retval
#
#     return rolling_apply1_(data, gsize, ngroups, mons)



################################################################################
#
# DATA GROUPING/FILTERING/TRIM/WINSORIZE
#
################################################################################

def _combine_array(array, by_array):
    if by_array is not None:
        return pd.DataFrame(np.array([array.values, by_array.values]).T)
    else:
        return array.reset_index(drop=True)


def _split_apply(array, fcn, *args):
    if array.ndim == 1:
        return fcn(array, *args)
    else:
        return fcn(array.iloc[:, 0], *args, array.iloc[:, 1])


def classify_array(array, split, ascending=True, by_array=None):
    """Classify array into classes.

    Class labels are set to 0, ..., no. quantiles-1, where 0 corresponds to the lowest (highest) group
    if `ascending` = True (False).
    If array contains nan, their classes are set to nan.

    Args:
        array: Nx1 ndarray or Series to be grouped.
        split: Number of classes or list of quantiles. (0.3, 0.7) is equivalent to (0.3, 0.7, 1.0).
        ascending (bool): Sorting order.
        by_array: Array based on which cut points are determined. If None, by_array = array. Eg, array can be set
            to ME and by_array to NYSE-ME to group firms on size with NYSE-size cut points.

    Returns:
        Nx1 ndarray or Series of classes. The output type corresponds to the input type.

    Examples:
        .. code-block::

            split = 10  # Classify array into 10 equally-spaced groups (0, ..., 9).
            split = [0.3, 0.7, 1.0]  # Classify array into three groups (0, 1, 2) that correspond to 0.3, 03-0.7, 0.7-1.0
                                     # quantiles.
            ascending = False  # Label 0 represents the largest group.
    """

    if isinstance(split, (int, np.integer)):  # number of classes
        quantiles = np.linspace(0, 1, split + 1)[1:]
    else:  # quantiles
        quantiles = split + ([1.0] if split[-1] < 1 else [])

    classes = np.full(array.shape, np.nan)

    if by_array is None:
        by_array = array
    by_array = by_array[~np.isnan(by_array)]

    if len(by_array) == 0:
        return pd.Series(classes, index=array.index) if type(array) == pd.Series else classes

    values = np.quantile(by_array, quantiles[:-1])
    classes[array > values[-1]] = 0
    for i, v in enumerate(values[::-1]):
        classes[array <= v] = i + 1

    if ascending:
        classes = len(quantiles) - 1 - classes

    return pd.Series(classes, index=array.index) if type(array) == pd.Series else classes


def classify(array, split, ascending=True, by_array=None):
    """ Classify `array` cross-sectionally.

    Class labels are set to 0, ..., no. quantiles-1, where 0 corresponds to the lowest (highest) group
    if `ascending` = True (False). If `array` contains nan, their classes are set to nan.

    Args:
        array: Data to classify. Series with date/id index.
        split: Number of classes or list of quantiles (see `classify_array`).
        ascending (bool): Sorting order.
        by_array: Data based on which quantile breakpoints are determined. If None, `by_array` = `array`.

    Returns:
        Series of class labels with index = `data.index`.

    Examples:
        Suppose `data` is a panel data that has columns 'me' (market equity) and 'me_nyse' (same as 'me' but
        has a value only when the stock is listed on NYSE).
        The following code groups stocks into terciles based on size with NYSE-size breakpoints and store the output
        in `data['me_class']`.
    ::

        split = [0.2, 0.7, 1.0]  # bottom 20%, 20-70%, top 30%
        data['me_class'] = classify(data['me'], split, by=data['me_nyse'])
    """

    array_ = _combine_array(array, by_array)
    classes = groupby_apply(array_.groupby(array.index.get_level_values(0)), _split_apply, classify_array, split, ascending)
    classes.sort_index(inplace=True)  # Restore the order of the rows.
    classes.index = array.index
    return classes


def _filter(array, n=None, q=None, ascending=True):
    v = array.values
    nancnt = int(np.isnan(v).sum())
    n = int(q * (len(v)-nancnt)) if q else n  # number of rows to keep

    idx = np.full(len(v), False)
    if n and ascending:
        idx[np.argpartition(v, min(len(v) - 1, n))[:n]] = True
    elif n:
        idx[np.argpartition(v, max(len(v) - n - nancnt, 0))[-n - nancnt:]] = True
        idx[np.isnan(v)] = False
    return pd.Series(idx, index=array.index)


def filter_n(data, on, n=None, q=None, ascending=True):
    """Filter `data` on `on`.

    If `n` is given, choose `n` smallest (if `ascending` = True) or largest (if `ascending` = False) within
    each date. If `q` is given, choose `q` quantile (from the bottom if `ascending` = True) within each date.

    Args:
        data: DataFrame or Series. The first index should be date.
        on: Column to filter on.
        n: Number of rows to keep.
        q: Quantile to keep.
        ascending (bool): If True, keep the smallest.

    Returns:
        Filtered data.

    Examples:
        Suppose `data` is a panel data (index = date/id) and has a column 'me' (market equity).
        The smallest 20% can be filtered out (keep the largest 80%) as follows:

        >>> data = filter(data, 'me', 0.8, ascending=False)
    """
    if (n is None) == (q is None):
        raise ValueError(f'Only one of n or q should be given.')

    array = data[on].reset_index(drop=True)
    idx = groupby_apply(array.groupby(data.index.get_level_values(0)), _filter, n, q, ascending)
    idx.sort_index(inplace=True)  # Restore the order of the rows.
    return data[idx.values]


def filter(data, on, limits, by=None):
    """Filter `data` on `on` with cut points determined by `by`.

    Filter out data if `data[on]` is below `limits[0]` quantile or above (1 - `limits[1]`) quantile.
    This function is similar to ``filter_n()`` but the cut points can be determined by another column values instead
    of the values of `on`.

    Args:
        data: DataFrame or Series. The first index should be date.
        on: Column to filter on.
        limits: Pair of quantiles, eg, (0.1, 0.1), (0.1, None).
        by: Column based on which cut points are determined. If None, `by` = `col`.

    Returns:
        Filtered data.

    Examples:
        Suppose `data` is a panel data (index = date/id) and has columns 'me' (market equity) and 'nyse_me'
        (market equity of NYSE stocks). The stocks whose sizes are smaller than the 0.2 NYSE-size quantile can be
        removed as follows:

        >>> data = filter(data, 'me', (0.2, None), by='nyse_me')
    """

    by = by or on
    array = data[on]
    by_array = data[by]

    array_ = _combine_array(array, by_array)

    idx = groupby_apply(array_.groupby(array.index.get_level_values(0)), _split_apply, _trim1d, limits)
    idx.sort_index(inplace=True)  # Restore the order of the rows.
    return data[idx.values]


def _trim1d(array, limits, by_array=None):
    if by_array is None:
        by_array = array
    by_array = by_array[~np.isnan(by_array)]

    lq = np.quantile(by_array, limits[0]) if limits[0] else None
    uq = np.quantile(by_array, 1-limits[1]) if limits[1] else None

    return (array > lq if lq else True) & (array < uq if uq else True)


def trim(array, limits, by_array=None):
    """Trim `array` within each date that are outside of the quantile values defined by `limits`.

    E.g., if `limits` = (0.1, 0.1), trim data below 0.1 quantile and above 0.9 quantile. The elements of `limits` can be
    set to None for one-sided trim, e.g., `limits` = (0.1, None).

    Args:
        array (Series): Data to trim. The first index should be date.
        limits: Pair of quantiles, eg, (0.1, 0.1), (0.1, None).
        by_array: Data based on which cut points are determined. If None, `by_array` = `array`.

    return:
        Trimmed data.
    """

    array_ = _combine_array(array, by_array)
    idx = groupby_apply(array_.groupby(array.index.get_level_values(0)), _split_apply, _trim1d, limits)
    idx.sort_index(inplace=True)  # Restore the order of the rows.
    return array[idx.values]


def _trim2d(data, limits):
    lq = data.quantile(limits[0]) if limits[0] else None
    uq = data.quantile(1-limits[1]) if limits[1] else None

    for col in data:  # Set out-of-bound values to nan.
        data.loc[(True if lq is None else data[col] < lq[col]) |
                 (True if uq is None else data[col] > uq[col]), col] = np.nan

    return data

def trim_data(data, limits):
    """Trim each column of `data` within each date that are outside of the quantile values defined by `limits`.

        The trimmed values are set to nan and the output shape is equal to the input shape.

    Args:
        data (DataFrame): Data to trim. The first index should be date.
        limits: Pair of quantiles, eg, (0.1, 0.1), (0.1, None).

    return:
        Trimmed data.
    """

    index = data.index
    data = data.reset_index(drop=True)
    data = groupby_apply(data.groupby(index.get_level_values(0)), _trim2d, limits)
    data.sort_index(inplace=True)  # Restore the order of the rows.
    data.index = index
    return data


def _winsorize(data, limits, by_array=None):
    if by_array is None:
        by_array = data
    lq = by_array.quantile(limits[0]) if limits[0] else None
    uq = by_array.quantile(1-limits[1]) if limits[1] else None

    if data.ndim == 1:
        return data.clip(lower=lq, upper=uq)
    else:
        return data.clip(lower=lq, upper=uq, axis=1)

def winsorize(array, limits, by_array=None):
    """Winsorize `array` within each date that are outside of the quantile values defined by `limits`.

    Args:
        array (Series): Data to winsorize. The first index should be date.
        limits: Pair of quantiles, eg, (0.1, 0.1), (0.1, None).
        by_array: Data based on which cut points are determined. If None, `by_array` = `array`.

    Returns:
        Winsorized data.
    """

    array_ = _combine_array(array, by_array)
    new_array = groupby_apply(array_.groupby(array.index.get_level_values(0)), _split_apply, _winsorize, limits)
    new_array.sort_index(inplace=True)  # Restore the order of the rows.
    new_array.index = array.index
    return new_array

def winsorize_data(data, limits):
    """Winsorize each column of `data` within each date that are outside of the quantile values defined by `limits`.

    Args:
        data (DataFrame): Data to winsorize.. The first index should be date.
        limits: Pair of quantiles, eg, (0.1, 0.1), (0.1, None).

    Returns:
        Winsorized data.
    """

    index = data.index
    data = data.reset_index(drop=True)
    data = groupby_apply(data.groupby(index.get_level_values(0)), _winsorize, limits)
    data.sort_index(inplace=True)  # Restore the order of the rows.
    data.index = index
    return data


################################################################################
#
# DATA INSPECTION/COMPARISON
#
################################################################################

def inspect_data(data, option=['summary'], date_col=None, id_col=None):
    """Inspect data.

    This function inspects a panel data, `data`, and print the results.

    * 'summary': `data` shape, number of unique dates, and number of unique ids.
    * 'id_count`: Number of ids per date.
    * 'nans': Number of nans and infs per column.
    * 'stats': Descriptive statistics. Same as ``data.describe()``.

    Args:
        data: Dataframe. It should have date and id columns or index = date/id.
        option: List of items to display. Available options:
                'summary', 'id_count', 'nans', 'stats'.
        date_col: Date column. If None, `date.index[0]` is assumed to be date.
        id_col: ID column. If None, `date.index[1]` is assumed to be id.
    """

    if date_col:
        dates = data[date_col]
    else:
        date_col = data.index.names[0]
        dates = data.index.get_level_values(0)

    if id_col:
        ids = data[id_col]
    else:
        ids = data.index.get_level_values(1)

    if 'summary' in option:
        log(f'Shape: {data.shape}')
        log(f'No. unique dates: {len(dates.unique())}, Date range: {dates.min()} - {dates.max()}')
        log(f'No. unique ids: {len(ids.unique())}')

    if 'id_count' in option:
        log(f'No. ids per date:\n {data.groupby(date_col).size()}')

    if 'nans' in option:
        log('\n')
        log('Nans, Infs')
        nans = data.isna().sum()
        nans = pd.concat([nans, nans / data.shape[0]], axis=1, keys=['no. nans', 'percentage'])

        infs = ((data == np.inf) | (data == -np.inf)).sum()
        infs = pd.concat([infs, infs / data.shape[0]], axis=1, keys=['no. infs', 'percentage'])

        log(pd.concat([nans, infs], axis=1), header=False)

    if 'stats' in option:
        log('\n')
        log('Descriptive statistics')
        log(data.describe(), header=False)


def compare_data(data1, data2=None, on=None, how='inner', tolerance=0.01, suffixes=('_x', '_y'), returns=False):
    """Compare `data1` with `data2`.

    This function compares the common columns of `data1` and `data2`.
    This is similar to ``data1.compare(data2)``, but `data1` and `data2` are not required to have the same index and
    columns. Also, a tolerance can be set to determine whether two values are the same.

    Args:
        data1: Dataframe for comparison.
        data2: Dataframe for comparison. If None, `data1` is assumed to be a merged dataset of `data1` and `data2`.
            If `data1` is a merged dataset, `on` and `how` have no effect.
        on: A column or a list of columns to merge data sets on. If None, data sets will be merged on index.
        how: How to merge: 'inner', 'outer', 'left', or 'right'
        tolerance: Tolerance level to determine equality. Two values, `val1` and `val2` are considered to be the same
            if `abs((val1 - val2) / val2) < threshold`.
        suffixes: suffixes to add to overlapping columns or suffixes used in the merged dataset.
            `suffixes[0]`: suffix for `data1`, `suffixes[1]`: suffix for `data2`.
        returns: If True, return the merged data.
    """

    if data2 is None:  # data1 is already a merged dataset.
        data = data1
    else:
        on = on or data1.index.names  # if None, merge on index.
        data = pd.merge(data1, data2, on=on, how=how, suffixes=suffixes)
        data.sort_index(axis=1, inplace=True)

    log(f"{'column': <20}  {'matched': >7}  {'corr': >7}", header=False)
    for col in data.columns:
        if (col[-2:] == suffixes[0]):
            if data[col].dtype not in ('float', 'float32', 'int', 'int64'):
                continue
            col2 = col[:-2] + suffixes[1]
            if col2 not in data:
                continue
            pair = data[[col, col2]]
            pair = pair[~pair.isin([np.nan, np.inf, -np.inf]).any(1)]  # drop nan or inf
            corr = pair[col].corr(pair[col2])  # np.corrcoef(pair.T)[0, 1]
            pair['diff'] = (pair[col] - pair[col2]).abs()
            matchidx = ((pair[col2].abs() < 1e-5) & (pair['diff'] < 1e-5)) | (pair['diff'] / pair[col2].abs() < tolerance)
            match = matchidx.sum() / len(pair)
            log(f'{col[:-2]: <20} {match: 7.5f} {corr: 7.5f}', header=False)

    if returns:
        return data


################################################################################
#
# AUX FUNCTIONS
#
################################################################################

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def fcn(k, g, freq, method, limit):
    if method == 'ffill':
        return k, g.resample(freq_map[freq]).pad(limit)
    else:  # fill with None
        return k, g.resample(freq_map[freq]).asfreq()


def populate(data, freq, method='ffill', limit=None):
    """Populate data.

    References:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html

    Args:
        data: Dataframe with index = date/id
        freq: Frequency to populate: QUARTERLY, MONTHLY, or DAILY.
        method: Filling method for newly added rows. 'ffill': forward fill, None: None.
        limit: Maximum number of rows to forward-fill.

    Returns:
        Populated data.
    """
    id_col = data.index.names[-1]

    gb = data.reset_index(level=id_col).groupby(id_col)
    retval = []
    futures = []
    with ProcessPoolExecutor() as executor:
        for k, g in gb:
            futures.append(executor.submit(fcn, k, g, freq, method, limit))

        for f in as_completed(futures):
            k, retval_ = f.result()
            retval_[id_col] = k
            retval.append(retval_)

    return pd.concat(retval).set_index(id_col, append=True).sort_index(level=[1,0])


def populate2(data, freq, method='ffill', limit=None):
    """Populate data.

    References:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html

    Args:
        data: Dataframe with index = date/id
        freq: Frequency to populate: QUARTERLY, MONTHLY, or DAILY.
        method: Filling method for newly added rows. 'ffill': forward fill, None: None.
        limit: Maximum number of rows to forward-fill.

    Returns:
        Populated data.
    """

    id_col = data.index.names[-1]

    gb = data.reset_index(level=id_col).groupby(id_col)
    retval = []
    for k, g in gb:
        if method == 'ffill':
            retval.append(g.resample(freq_map[freq]).pad(limit))
        else:  # fill with None
            retval.append(g.resample(freq_map[freq]).asfreq())
        retval[-1][id_col] = k

    return pd.concat(retval).set_index(id_col, append=True)

    # The code below is simpler but three times slower.
    # if method == 'ffill':
    #     pop_data = data.reset_index(level=id_col).groupby(id_col).resample(freq_map[freq]).pad(limit).swaplevel()
    # else:  # fill with None
    #     pop_data = data.reset_index(level=id_col).groupby(id_col).resample(freq_map[freq]).asfreq().swaplevel()
    # pop_data.drop(columns=id_col, inplace=True)
    #
    # return pop_data


def to_month_end(date):
    """Shift dates to the last dates of the same month.

    Args:
        date: datetime Series.

    Returns:
        datetime Series shifted to month end.
    """

    return date + MonthEnd(0)


def add_months(date, months, to_month_end=True):
    """Add months to dates.

    Args:
        date: datetime Series.
        months: Months to add. Can be negative.
        to_month_end: If True, returned dates are end-of-month dates.

    Returns:
        datetime Series of (`date` + `months`). Dates are end-of-month dates if `to_month_end` = True.
    """

    if to_month_end:
        return date + MonthEnd(months)
    else:
        return date + pd.DateOffset(months=months)




