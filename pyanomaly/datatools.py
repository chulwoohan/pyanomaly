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
def parse_return_(retval):
    """Parse returns from groupby_apply or groupby_apply_np.

    Args:
        retval: returns from groupby_apply or groupby_apply_np. retval should be a list of one of the followings:
        i) (tuple) of float
        ii) (tuple) of ndarray
        iii) (tuple) of dataframe/series.

    Returns:
        Concatenated value of retval. The number of returns is equal to the number of returns by fcn in groupby_apply.
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
    """Apply fcn to groups in gb. The first input of fcn should the dataframe of a group.

    Args:
        gb: pd.GroupBy
        fcn(gbdata, *args): function to apply to groups. The first argument should be the grouped data.

    Returns:
        Concatenated function outputs
    """
    retval = [fcn(v, *args) for k, v in gb]
    return parse_return_(retval)


def rolling_apply(data, by=None, fcn=None, *args):
    """Apply fcn to each group grouped by 'by. If by=None, group by the last index, i.e. same as groupby(level=-1)."""
    by = by or data.index.names[-1]
    return groupby_apply(data.groupby(by), fcn, *args)


def groupby_apply_np(gb, fcn, *args):
    """Apply fcn to groups in gb. The input of fcn should be the values of the dataframe of a group.

    Args:
        gb: pd.GroupBy
        fcn(gbdata, *args): function to apply to groups. The first argument should be the values of the grouped data.

    Returns:
        Concatenated function outputs
    """

    # retval = []
    # for k, v in gb:
    #     if k == 14535:
    #         print(k)
    #     retval.append(fcn(v.values, *args))
    retval = [fcn(v.values, *args) for k, v in gb]
    return parse_return_(retval)


def rolling_apply_np(data, by=None, fcn=None, *args):
    """Apply fcn to each group grouped by 'by. If by=None, group by the last index, i.e. same as groupby(level=-1)."""

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
# DATA FILTERING/TRIM/WINSORIZE
#
################################################################################
def data_filter_(data, q=None, n=None, ascending=True):
    data = data.dropna()
    n = int(q * len(data)) if q else n
    idx = np.array([False] * len(data))
    if n and ascending:
        idx[np.argpartition(data.values, min(len(data) - 1, n))[:n]] = True
    elif n:
        idx[np.argpartition(data.values, max(len(data) - n, 0))[-n:]] = True
    return data.loc[idx]


def data_filter(data, col, q=None, n=None, ascending=True):
    """Filter data on col. If n is given, choose n smallest (if ascending=True) or largest (if ascending=False) within
    each date. If q is given, choose q quantile (from the bottom if ascending=True) within each date.

    Args:
        data: DataFrame. first index should be date.
        col: column to filter on
        q: quantile to keep
        n: number of rows to keep
        ascending: if True, keep the smallest.

    return:
        Filtered data.
    """
    if (q is None) == (n is None):
        raise ValueError(f'Only one of p and n should be given.')

    index = data.index
    data.reset_index(drop=True, inplace=True)
    new_data = groupby_apply(data.groupby(index.get_level_values(0))[col], data_filter_, q, n, ascending)
    new_data.sort_index(inplace=True)  # Restore the order of the rows.
    new_data.index = index
    return new_data

    # new_data = groupby_apply(data.groupby(level=0)[col], data_filter_, q, n, ascending)
    # return new_data.loc[data.index]


def trim_(data, limits, copy=True, keep_row=False):
    retval = data.copy() if copy else data

    if data.ndim == 1:
        array = data.dropna()
        if limits[0]:
            v = np.quantile(array, limits[0])
            if keep_row:
                retval[retval < v] = None
            else:
                retval = retval[retval >= v]
        if limits[1]:
            v = np.quantile(array, 1 - limits[1])
            if keep_row:
                retval[retval > v] = None
            else:
                retval = retval[retval <= v]
    else:
        for col in data.columns:
            array = data[col].dropna()
            if limits[0]:
                v = np.quantile(array, limits[0])
                retval.loc[retval[col] < v, col] = None
            if limits[1]:
                v = np.quantile(array, 1 - limits[1])
                retval.loc[retval[col] > v, col] = None
    return retval


def trim(data, limits, copy=True, keep_row=False):
    """Trim data within each date that are outside of the quantile values defined by limits.
    Eg: If limits = (0.1, 0.1), trim data below 0.1 quantile and above 0.9 quantile. The elements of limits can be set
    to None for one-sided trim, e.g., limits = (0.1, None).

    Args:
        data: DataFrame or Series. first index should be date.
        limits: pair of quantiles, eg, (0.1, 0.1), (0.1, NOne).
        copy: if False, overwrite data instead of creating a new data. False will still make a copy if keep_row=False.
        keep_row: If True, the trimmed data will be set to None and the shape of data is maintained. If False, trimmed
        data will be removed. If data has multiple columns, keep_row is ignored (set to True always).

    return:
        Trimmed data.
    """
    index = data.index
    data.reset_index(drop=True, inplace=True)
    new_data = groupby_apply(data.groupby(index.get_level_values(0)), trim_, limits, copy, keep_row)
    new_data.sort_index(inplace=True)  # Restore the order of the rows.
    new_data.index = index
    return new_data


def winsorize_(data, limits, copy=True):
    # retval = data.copy() if copy else data

    l = limits[0] or 0
    u = 1 - (limits[1] or 0)
    retval = data.clip(lower=data.quantile(l), upper=data.quantile(u))
    # if data.ndim == 1:
    #     retval[:] = mstats.winsorize(data, limits=limits, nan_policy='omit').data
    # else:
    #     for col in data.columns:
    #         retval[col] = mstats.winsorize(data[col], limits=limits, nan_policy='omit').data

    return retval


def winsorize(data, limits, copy=True):
    """Winsorize each column of data.
    Args:
        data: DataFrame or Series. first index should be date.
        limits: pair of quantiles, eg, (0.1, 0.1), (0.1, NOne).
        copy: if False, overwrite data instead of creating a new data.

    Returns:
        Winsorized data.
    """
    index = data.index
    data.reset_index(drop=True, inplace=True)
    new_data = groupby_apply(data.groupby(index.get_level_values(0)), winsorize_, limits, copy)
    new_data.sort_index(inplace=True)  # Restore the order of the rows.
    new_data.index = index
    return new_data

    # Simpler, but much slower method.
    # new_data = groupby_apply(data.groupby(level=0), winsorize_, limits, copy)
    # return new_data.loc[data.index]


################################################################################
#
# DATA INSPECTION/COMPARISON
#
################################################################################

def inspect_data(data, option=['summary'], date_col=None, id_col=None):
    """Inspect data.

    Args:
        data: Dataframe with index=date/id
        option: list of items to display. Available options:
                'summary', 'id_count', 'nans', 'stats'
    """
    if date_col:
        dates = data[date_col]
    else:
        date_col = data.index.names[0]
        dates = data.index.get_level_values(0)

    if id_col:
        ids = data[id_col]
    else:
        id_col = data.index.names[1]
        ids = data.index.get_level_values(1)

    if 'summary' in option:
        log(f'Shape: {data.shape}')
        log(f'No. unique dates: {len(dates.unique())}, Date range: {dates.min()} - {dates.max()}')
        log(f'No. unique ids: {len(ids.unique())}')

    if 'id_count' in option:
        log(f'Average ids per date:\n {data.groupby(date_col).size()}')

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


def compare_data(data1, data2=None, on=None, how='inner', threshold=0.1):
    """Compare data1 with data2.
    Args:
        data1: Dataframe for comparison.
        data2: Dataframe for comparison. Data can be merged outside by setting data1=merged data, data2=None
        on: a column or a list of columns to merge data on. on=None will used index to merge.
        how: 'inner', 'outer', 'left', 'right'
        threshold: Consider matched if abs(val1 - val2 /val2) < threshold
    """
    if data2 is None:  # data1 is already a merged dataset.
        data = data1
    else:
        on = on or data1.index.names  # if None, merge on index.
        data = pd.merge(data1, data2, on=on, how=how, suffixes=('_a', '_b'))
        data.sort_index(axis=1, inplace=True)

    log(f"{'column': <20}  {'matched': >7}  {'corr': >7}", header=False)
    for col in data.columns:
        if (col[-2:] == '_a'):
            if data[col].dtype not in (np.float, np.int):
                continue
            col2 = col[:-1] + 'b'
            if col2 not in data:
                continue
            pair = data[[col, col2]]
            pair = pair[~pair.isin([np.nan, np.inf, -np.inf]).any(1)]  # drop nan or inf
            corr = pair[col].corr(pair[col2])  # np.corrcoef(pair.T)[0, 1]
            pair['diff'] = (pair[col] - pair[col2]).abs()
            matchidx = ((pair[col2].abs() < 1e-5) & (pair['diff'] < 1e-5)) | (pair['diff'] / pair[col2].abs() < threshold)
            match = matchidx.sum() / len(pair)
            log(f'{col[:-2]: <20} {match: 7.5f} {corr: 7.5f}', header=False)

    return data

################################################################################
#
# AUX FUNCTIONS
#
################################################################################

def populate(data, freq, method='ffill', limit=None):
    """Populate data.

    Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html

    Args:
        data: Dataframe with index=date/id
        freq: frequency to populate. Eg: MONTHLY, DAILY.
        method: filling method for newly added rows. 'ffill': forward fill, None: None.
        limit: Max number of rows to forward-fill.

    Returns:
        Populated data.
    """
    id_col = data.index.names[-1]
    if method == 'ffill':
        pop_data = data.reset_index(level=id_col).groupby(id_col).resample(freq_map[freq]).pad(limit).swaplevel()
    else:  # fill with None
        pop_data = data.reset_index(level=id_col).groupby(id_col).resample(freq_map[freq]).asfreq().swaplevel()
    pop_data.drop(columns=id_col, inplace=True)

    return pop_data


def to_month_end(date):
    """Shift date to the last date of the same month.

    Args:
        date: Series of datetime.

    Returns:
        Series of datetime shifted to month end.
    """
    return date + MonthEnd(0)


def add_months(date, months, to_month_end=True):
    """Add months.

    Args:
        date: datetime Series.
        months: months to add. Can be negative.
        to_month_end: If True, returned dates are end-of-month dates.

    Returns:
        datetime Series of (date + months). Dates are end-of-month dates if to_month_end=True.
    """
    if to_month_end:
        return date + MonthEnd(months)
    else:
        return date + pd.DateOffset(months=months)


def make_target_return(ret, period=1):
    """Compute future returns.

    Args:
        ret: Series of returns.
        period: target period. Eg: If period=3, 3-period ahead (cumulative) returns.

    Returns:
        Series of target returns.
    """
    if period == 1:
        return ret.shift(-1)
    else:
        return np.exp(np.log(ret + 1).shift(-period).rolling(period).sum()) - 1




