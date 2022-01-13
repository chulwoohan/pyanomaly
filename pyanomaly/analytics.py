"""This module defines analytic functions.

Sorting
    * ``one_dim_sort()``: One-dimensional sort.
    * ``two_dim_sort()``: Two-dimensional sort.

Time-Series Analysis
    * ``time_series_average()``: Calculate time-series mean and t-value.
    * ``rolling_beta()``: Run rolling OLS on a panel data.

Cross-sectional Analysis
    * ``crosssectional_regression()``: Run cross-sectional OLS and calculate the time-series means and t-values of the coefficients.

Portfolio Analysis
    * ``make_future_return()``: Compute future returns.
    * ``weighted_mean()``: Calculate weighted mean.
    * ``make_quantile_portfolios()``: Make quantile portfolios.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pyanomaly.globals import *
from pyanomaly.numba_support import *
from pyanomaly.datatools import *
from pyanomaly.portfolio import Portfolio, Portfolios
from pyanomaly.multiprocess import multiprocess

################################################################################
#
# ONE-DIM / TWO-DIM SORT
#
################################################################################

def relabel_class(data, labels=None, axis=0, level=-1, col=None):
    """Relabel classes in columns/indexes or in a column of `data`.

    The existing labels should be continuous integers starting from 0.
    The `data` is relabeled in-place.

    Args:
        data: DataFrame to be relabeled.
        labels: New class labels.
        axis: 1: index, 2: column.
        level: Level of index/column to be relabeled.
        col: Column name. If column name is given, `axis` and `level` are ignored.
    """

    if labels is None:
        return

    name_mapping = {i: l for i, l in enumerate(labels)}

    if col is None:
        data.rename(name_mapping, axis=axis, level=level, inplace=True)
    else:
        for i, l in enumerate(labels):
            data.loc[data[col] == i, col] = l


def append_long_short(data, level=-1, l_label=None, s_label=None, ls_label=None):
    """Add long-short to `data`.

    Long-short is defined as (first group - last group) in each date.
    If labels are not given, long-short will be (class 0 - class N-1), where N is the number of classes,
    and the class label of the long-short is set to N.

    Args:
        data: DataFrame with index = date/class1/class2/....
        level: Index level to make long-short on. Default to the last level.
        l_label: Label of the long class. If None, `l_label = 0`.
        s_label: Label of the short class. If None, 's_label = num. classes - 1`.
        ls_label: Label of the long-short: If None, `ls_label = num. classes`.

    Returns:
        `data` with long-short appended.
    """

    data2 = data.swaplevel(-1, level).unstack()
    classes = list(data2.columns.get_level_values(-1).unique())
    l_label = l_label or 0
    s_label = s_label or len(classes) - 1
    ls_label = ls_label or len(classes)

    for col in data:
        if isinstance(col, tuple):
            data2[(*col, ls_label)] = data2[(*col, l_label)] - data2[(*col, s_label)]
        else:
            data2[(col, ls_label)] = data2[(col, l_label)] - data2[(col, s_label)]

    return data2.stack(dropna=True).swaplevel(-1, level)[data.columns]

# def append_long_short(data):
#
#     def append_long_short_(data):
#         date = data.index[0][0]
#         if len(data.index.levshape) == 3:
#             class1 = data.index[0][1]
#             ls = data.loc[(date, class1, l_label)] - data.loc[(date, class1, s_label)]
#             ls.name = (date, class1, ls_label)
#         else:
#             ls = data.loc[(date, l_label)] - data.loc[(date, s_label)]
#             ls.name = (date, ls_label)
#         return data.append(ls)
#
#     n_classes = len(data.index.get_level_values(-1).unique())
#     l_label = 0
#     s_label = n_classes - 1
#     ls_label = n_classes
#
#     return groupby_apply(data.groupby(level=0), append_long_short_)


def one_dim_sort(data, class_col, target_cols=None, weight_col=None, function='mean', add_long_short=True):
    """One-dimensional sort.

    This function assumes that `data` has already been sorted/grouped and the class labels are given
    in `class_col` column. Aggregate `data[target_cols]` using `data[class_col]` and return aggregated results.

    Args:
        data: DataFrame to be grouped. Index must be date/id.
        class_col: Class label column.
        target_cols: (List of) column(s) to aggregate. If None, `target_cols` = all numeric columns of `data`.
        weight_col: Weight column. If given, weighted mean is returned.
        function: Aggregate function, e.g., 'sum', 'mean', 'count', or a list of functions.
            If `function` != 'mean', `weight_col` is ignored.
        add_long_short (bool): Add long-short to the output.

    Returns:
        Aggregated data with index=date/class, columns = `target_cols`.
    """

    date_col = data.index.names[0]
    if target_cols is None:  # target_cols = all columns in data except for non-numeric ones.
        target_cols = list(data.columns[(data.dtypes == float) | (data.dtypes == int)])
    else:
        target_cols = to_list(target_cols)

    if (function == 'mean') and weight_col:  # value_weight
        ret_data = weighted_mean(data, target_cols, weight_col, [date_col, class_col])
    elif isinstance(function, str):  # single function
        gb = data.groupby([date_col, class_col])[target_cols]
        ret_data = eval(f'gb.{function}()')
    else:  # list of functions
        gb = data.groupby([date_col, class_col])[target_cols]
        ret_data = eval(f'gb.agg({function})')

    # Remove unclassified data.
    # ret_data = ret_data[~ret_data.index.get_level_values(class_col).isin([np.nan, ''])]

    if add_long_short:
        ret_data = append_long_short(ret_data)

    return ret_data


def two_dim_sort(data, class_col1, class_col2, target_cols=None, weight_col=None, function='mean',
                 add_long_short=True, output_dim=1):
    """Two-dimensional sort.

    This function assumes that `data` has already been sorted/grouped and the class labels are given
    in `class_col1` and `class_col2` columns. Aggregate `data[target_cols]` using `data[class_col1, class_col2]` and
    return aggregated results.

    Args:
        data: Data to be grouped. Index must be date/id.
        class_col1/2: Class label column for the 1st (2nd) dimension.
        target_cols: (List of) column(s) to aggregate. If None, `target_cols` = all numeric columns of `data`.
        weight_col: Weight column. If given, weighted mean is returned.
        function: Aggregate function, e.g., 'sum', 'mean', 'count'. If it is other than 'mean', `weight_col` is ignored.
        add_long_short (bool): Add long-short to the output.
        output_dim: If 1, output is a DataFrame with index=date/class1/class2; if 2, output is a DataFrame with
            index=date/class1 and column=class2.

    Returns:
        Aggregated data (DataFrame or dict of DataFrame).

        * If `output_dim` = 1, index = date/class1/class2, columns = `target_cols`
        * If `output_dim` = 2 and `len(target_cols)` = 1, index = date/class1, columns = class2
        * If `output_dim` = 2 and `len(target_cols)` > 1, output is dict with keys = `target_cols` and
            values = DataFrames (as in the second case).
    """

    date_col = data.index.names[0]
    if target_cols is None:  # target_cols = all columns in data except for non-numeric ones.
        target_cols = list(data.columns[(data.dtypes == float) | (data.dtypes == int)])
    else:
        target_cols = to_list(target_cols)

    if (function == 'mean') and weight_col:  # value_weight
        ret_data = weighted_mean(data, target_cols, weight_col, [date_col, class_col1, class_col2])
    else:  # equal-weight
        # gb = data[[class_col1, class_col2] + target_cols].groupby([date_col, class_col1, class_col2])
        gb = data.groupby([date_col, class_col1, class_col2])[target_cols]
        ret_data = eval(f'gb.{function}()')

    # Remove unclassified data.
    # ret_data = ret_data[~ret_data.index.get_level_values(class_col1).isin([np.nan, ''])]
    # ret_data = ret_data[~ret_data.index.get_level_values(class_col2).isin([np.nan, ''])]

    if add_long_short:
        ret_data = append_long_short(ret_data, level=-2)
        ret_data = append_long_short(ret_data, level=-1)

    if output_dim == 1:  # 1D with index=date/class1/class2
        return ret_data  # .to_frame(name=target_col)
    else:  # 2D with index=date/class1, column=class2
        index = ret_data.index.get_level_values(-2).unique()
        columns = ret_data.index.get_level_values(-1).unique()
        if len(target_cols) == 1:
            return ret_data[target_cols[0]].unstack().reindex(index, level=-1)[
                columns]  # this is necessary as unstack() sort index and columns.
        else:
            ret_dict = {}
            for target_col in target_cols:
                # Below, reindex is necessary as unstack() sort index and columns.
                ret_dict[target_col] = ret_data[target_col].unstack().reindex(index, level=-1)[columns]
            return ret_dict


################################################################################
#
# TIME-SERIES ANALYSIS
#
################################################################################

def t_value(data, cov_type='nonrobust', cov_kwds=None):
    """Calculate t-value for each column under H0: x = 0.

    Args:
        data: DataFrame with each column containing samples.
        cov_type: See `sm.OLS.fit()`, eg, 'HAC' for Newey-West.
        cov_kwds: See `sm.OLS.fit()`, eg, {'maxlags: 12} for cov_type = 'HAC'.

    Returns:
        Series of t-values with index = `data.columns`.
    """

    x = np.ones([len(data), 1])
    if isinstance(data, pd.Series):
        est = sm.OLS(data, x, missing='drop').fit(cov_type=cov_type, cov_kwds=cov_kwds)
        return est.tvalues[0]
    else:  # DataFrame
        res = {}
        for col in data.columns:
            est = sm.OLS(data[col], x, missing='drop').fit(cov_type=cov_type, cov_kwds=cov_kwds)
            res[col] = est.tvalues[0]
        res = pd.Series(res)
        res.index = res.index.set_names(data.columns.names)
        return res


def time_series_average(data, cov_type='nonrobust', cov_kwds=None):
    """Calculate time-series mean and t-value of each column (and for each group).

    `data` can be either a time-series data (index=date) or a panel data (index=date/group).
    If it is a panel, time-series mean and t-value are calculated for each group.

    Args:
        data (DataFrame): Data to analyze. If MultiIndex, the first index must be date.
        cov_type: See `sm.OLS.fit()`, eg, 'HAC' for Newey-West.
        cov_kwds: See `sm.OLS.fit()`, eg, {'maxlags: 12} for cov_type = 'HAC'.

    Returns:
        mean (DataFrame), t-value (DataFrame).

        * If data has MultiIndex, mean (t-value) has index = `data.index[1:]`, columns = `data.columns`.
        * Otherwise, mean (t-value) has index = `data.columns`, columns = 'mean' ('t-stat').
    """
    if not isinstance(data.index, pd.MultiIndex):  # single (date) index
        mean = data.mean().to_frame(name='mean')
        tval = t_value(data, cov_type, cov_kwds).to_frame(name='t-stat')
        return mean, tval
    else:  # MultiIndex.
        data = data.reset_index(level=0, drop=True)
        index = data.index.unique()

        mean = data.groupby(index.names).mean()
        tval = data.groupby(index.names).apply(t_value, cov_type, cov_kwds)
        if isinstance(data, pd.DataFrame):
            tval.columns.names = mean.columns.names
        return mean.reindex(index), tval.reindex(index)  # reindex to keep the order of index


def rolling_beta(data, window, minobs=None, endo_col=None, exog_cols=None):
    """Run rolling OLS on a panel, `data`.

    `data` must have index=date/id and Rolling OLS is applied to each id.
    This is faster than `statsmodels.RollingOLS`, but the output is limited to
    coefficients, R2, and idiosyncratic volatility.

    Args:
        data: DataFrame with index = date/id, sorted on id/date.
        window: Window size.
        minobs: Minimum number of observations in the window. If observations < `minobs`, result is nan.
        endo_col: y column.
        exog_cols: X columns (without constant). If `endo_col = exog_cols = None`, `endo_col = data[:, 0]`,
            `exog_cols = data[:, 1:]`.

    Returns:
        Coefficients, R2, idiosyncratic volatility. These are NxK, Nx1, and Nx1 ndarrays, respectively,
        where N = `len(data)` and K = `len(exog_cols)+1`.
    """

    minobs = minobs or window
    data = data if endo_col is None else data[[endo_col] + exog_cols]

    @njit(error_model='numpy')
    def fcn(data):
        beta = np.full(data.shape, np.nan)
        r2 = np.full((data.shape[0], 1), np.nan)
        idio = np.full((data.shape[0], 1), np.nan)

        isnan = isnan1(data)
        for i in range(minobs - 1, data.shape[0]):
            i0 = i - min(i, window-1)
            sample = data[i0:i + 1]
            sample = sample[~isnan[i0:i + 1]]

            if sample.shape[0] < minobs:
                continue

            endo = sample[:, 0].copy()
            exog = add_constant(sample[:, 1:])

            try:
                beta[i, :] = (np.linalg.solve(exog.T @ exog, exog.T @ endo)).reshape(data.shape[1])
                pred = (beta[i, :] * exog).sum(1)
                r2[i] = 1 - ((endo - pred) ** 2).sum() / ((endo - endo.mean()) ** 2).sum()
                idio[i] = std(endo - pred)
            except:
                continue

        return beta, r2, idio

    return rolling_apply_np(data, fcn=fcn)



################################################################################
#
# CROSS-SECTIONAL ANALYSIS
#
################################################################################

def crosssectional_regression(data, endo_col, exog_cols, add_constant=True, cov_type='nonrobust', cov_kwds=None):
    """Run cross-sectional OLS on each date and calculate the time-series means and t-values of the coefficients.

    Args:
        data: DataFrame with index=date/id
        endo_col: y column.
        exog_cols: List of X columns.
        add_constant (bool): Add constant to x.
        cov_type: See `sm.OLS.fit()`, eg, 'HAC' for Newey-West.
        cov_kwds: See `sm.OLS.fit()`, eg, {'maxlags: 12} for cov_type = 'HAC'.

    Returns:
        mean, tval, coefs.

        * mean (DataFrame): Time-series means of coefficients with index=('const') + `exog_cols`, columns='mean'.
        * tval (DataFrame): t-values of coefficients with index=('const') + `exog_cols`, columns='t-stat'.
        * coefs (DataFrame): Coefficient time-series with index=dates, columns=('const') + `exog_cols`.
    """
    data = data.dropna(subset=[endo_col] + exog_cols)

    # cross-sectional regression
    coefs = {}
    for k, v in data.groupby(level=0):
        y = v[endo_col]
        x = sm.add_constant(v[exog_cols]) if add_constant else v[exog_cols]
        if x.shape[0] <= x.shape[1]:  # If not enough sample for regression, skip.
            continue
        model = sm.OLS(y, x).fit()

        coefs[k] = model.params
    coefs = pd.DataFrame(coefs).T

    # time series means and t-values of the cross-sectional regression coefficients
    mean, tval = time_series_average(coefs, cov_type, cov_kwds)

    return mean, tval, coefs


################################################################################
#
# Portfolio Analysis
#
################################################################################

def make_future_return(ret, period=1):
    """Compute future returns.

    This is a simple function to compute `period`-period ahead return.

    See Also:
        ``Panel.cumret()``

    Args:
        ret: Series of returns with index = date or date/id.
        period: target period. Eg: If period=3, 3-period ahead (cumulative) returns. If `ret` contains monthly (daily)
            returns, the output will be 3-month (day) ahead returns.

    Returns:
        Series of future returns.
    """

    if period == 1:
        futret = ret.shift(-1)
    else:
        futret = np.exp(np.log(ret + 1).shift(-period).rolling(period).sum()) - 1

    if isinstance(ret.index, pd.MultiIndex):
        ids = ret.index.get_level_values(-1).values
        futret[ids != np.roll(ids, -period)] = np.nan

    return futret


def weighted_mean(data, target_cols, weight_col, group_cols):
    """Calculate weighted means of `data[target_cols]` for each group defined by `group_cols`.

    Args:
        data: DataFrame.
        target_cols: (List of) column(s) to calculate weighted-mean.
        weight_col: Weight column name, or Series or ndarray of weights.
        group_cols: (List of) grouping column(s).

    Returns:
        DataFrame of weighted mean with index = `group_cols`, columns = `target_cols`.

    """

    target_cols = to_list(target_cols)
    group_cols = to_list(group_cols)
    if is_iterable(weight_col):
        weight = weight_col
    else:
        weight = data[weight_col]

    numer = data[target_cols].mul(weight, axis=0)
    denom = pd.notnull(data[target_cols]).mul(weight, axis=0)  # set weight=0 where target data is null.
    da = data[data.columns.intersection(group_cols)]  # data necessary for groupby (to save memory)
    gb1 = pd.concat([da, numer], axis=1).groupby(group_cols)
    gb2 = pd.concat([da, denom], axis=1).groupby(group_cols)

    return gb1[target_cols].sum().divide(gb2[target_cols].sum())


# def wmean(x, w):
#     if x.ndim == 1:
#         return x.dot(w) / w[~np.isnan(x)].sum()
#     elif x.ndim == 2:
#         retval = np.full(x.shape[1], np.nan)
#         for i in range(x.shape[1]):
#             x_ = x[:, i]#.copy()
#             retval[i] = x_.dot(w) / w[~np.isnan(x_)].sum()
#
#         return retval
#
#
# def weighted_mean(data, target_cols, weight_col, group_cols):
#     """Calculate weighted means of `data[target_cols]` for each group defined by `group_cols`.
#
#     Args:
#         data: DataFrame.
#         target_cols: (List of) column(s) to calculate weighted-mean.
#         weight_col: Weight column name, or Series or ndarray of weights.
#         group_cols: (List of) grouping column(s).
#
#     Returns:
#         DataFrame of weighted mean with index = `group_cols`, columns = `target_cols`.
#
#     """
#
#     if is_iterable(weight_col):
#         data['__weight__'] = weight_col
#         weight_col = '__weight__'
#
#     index = []
#     wmeans = []
#     columns = to_list(group_cols) + to_list(target_cols) + [weight_col]
#     columns = list(data.columns.intersection(columns))
#     for k, g in data[columns].groupby(group_cols):
#         index.append(k)
#         wmeans.append(wmean(g[target_cols].values, g[weight_col].values))
#
#     values = np.concatenate(wmeans) if isinstance(wmeans[0], np.ndarray) else wmeans
#     index = pd.MultiIndex.from_tuples(index, names=group_cols) if is_iterable(group_cols) and len(group_cols) > 1 else pd.Index(index, name=group_cols)
#     retval = pd.DataFrame(values, columns=to_list(target_cols), index=index)
#     if '__weight__' in data:
#         del data['__weight__']
#
#     return retval


def make_position(data, ret_col, weight_col=None, pf_col=None, other_cols=None):
    """Make portfolio position data from a panel data.

    To construct and evaluate a portfolio using ``Portfolio`` class, position data is required.
    This function makes the position data from the input data, which is a panel of securities.
    The position data is generated via the following operations:

        - Shift dates forward so that the return at t is the return between between t-1 and t,
          and other variables are as of t-1.
        - Change column names as assumed in ``Portfolio``.
            - 'date': Date column.
            - 'id': Security id column.
            - 'ret': Return column.
            - 'wgt': Weight column.
        - Generate portfolio weights by normalizing market equity.

    Args:
        data: DataFrame with index = date/id.
        ret_col: Return column of `data`.
        weight_col: Weight column of `data`. If None, constituents are equally weighted.
        pf_col: Portfolio column of `data`, i.e., a column with portfolio id (name) a security belongs to.
            This can be None if the input data is for one portfolio.
        other_cols: Other columns of `data` the user wants to keep in the position data.

    Returns:
        Position DataFrame with index = 'date' and columns = ['id', 'ret', 'wgt'] + `other_cols`.
    """

    columns = unique_list([ret_col, weight_col, pf_col, other_cols])
    position = data[columns].groupby(level=-1).shift(1)  # Shift data forward.
    position = position.dropna(subset=[ret_col])  # Drop nan returns.

    # Rename: date: 'date', id: 'id', ret_col: 'ret', weight_col: 'wgt'
    position.index = position.index.set_names(['date', 'id'])
    position.rename(columns={ret_col: 'ret'}, inplace=True)

    if weight_col is None:  # Equal-weight portfolio.
        position['wgt'] = 1.
    else:
        if other_cols and (weight_col in other_cols):  # keep weight_col.
            position['wgt'] = position[weight_col]
        else:  # rename weight_col.
            position.rename(columns={weight_col: 'wgt'}, inplace=True)

    if pf_col:  # Normalize weights per date/portfolio.
        position['wgt'] = position.wgt / position.groupby(['date', pf_col]).wgt.transform('sum')
    else:  # Normalize weights per date.
        position['wgt'] = position.wgt / position.groupby('date').wgt.transform('sum')

    return position.reset_index('id').sort_index()


def make_portfolio(data, ret_col, weight_col=None, costfcn=None, keep_position=True, name=''):
    """Construct a portfolio from panel data.

    This function creates portfolio position data from `data` and construct a portfolio from it.

    Args:
        data: DataFrame with index = date/id.
        ret_col: Return column of `data`.
        weight_col: Weight column of `data`. If None, constituents are equally weighted.
        costfcn: Transaction cost. See ``pyanomaly.Portfolio`` for details.
        keep_position: If True, keep the position information. If position information is not needed,
            set this to False to save memory.
        name: Portfolio name.

    Returns:
        ``Portfolio`` object.
    """

    position = make_position(data, ret_col, weight_col)
    return Portfolio(name, position, costfcn=costfcn, keep_position=keep_position)


def make_long_short_portfolio(lposition, sposition, ls_wgt=(1, -1), rf=None, costfcn=None, keep_position=True, name='H-L'):
    """Make a long-short portfolio.

    Args:
        lposition (DataFrame): Long position data.
        sposition (DataFrame): Short position data.
        ls_wgt: Long-short weights. (1, -1) means 1:1 long-short. If None, long-short portfolio is not constructed.
        rf: Series or DataFrame of risk-free rates. The index should be date.
        costfcn: Transaction cost. See ``pyanomaly.Portfolio`` for details.
        keep_position: If True, keep the position information. If position information is not needed,
            set this to False to save memory.
        name: Portfolio name.

    Returns:
        ``Portfolio`` object.
    """
    # Copy data as their values ('wgt') are changed.
    lposition = lposition.copy()
    sposition = sposition.copy()

    lposition['wgt'] *= ls_wgt[0]
    sposition['wgt'] *= ls_wgt[1]
    position = pd.concat([lposition, sposition]).sort_index()
    portfolio = Portfolio(name, position, rf=rf, costfcn=costfcn, keep_position=keep_position)
    return portfolio


def make_quantile_portfolios(position, pf_col, ls_wgt=(1, -1), rf=None, costfcn=None, keep_position=True, labels=None):
    """Make quantile portfolios.

    This function makes quantile portfolios and the long-short portfolio from position data.

    Args:
        position (DataFrame): Position data. See ``make_position()`` for the format.
        pf_col: Column of `position` that maps a security with quantiles (portfolios).
        ls_wgt: Long-short weights. (1, -1) means 1:1 long-short. If None, long-short portfolio is not constructed.
        rf: Series or DataFrame of risk-free rates. The index should be date.
        costfcn: Transaction cost. See ``pyanomaly.Portfolio`` for details.
        keep_position: If True, keep the position information. If position information is not needed,
            set this to False to save memory.
        labels: Portfolio names. If None, the values in `pf_col` are used.

    Returns:
        ``Portfolios`` object.
    """

    log(f'Making quantile portfolios for {pf_col}...')

    n_class = len(position[pf_col].unique())
    if position[pf_col].isna().any():
        n_class -= 1  # Exclude unclassified (nan) data.

    portfolios = Portfolios()
    for cls in range(n_class):
        pf_name = str(cls) if labels is None else labels[cls]
        cposition = position.loc[position[pf_col] == cls, position.columns.difference([pf_col])]
        portfolio = Portfolio(pf_name, cposition, rf=rf, costfcn=costfcn, keep_position=keep_position)
        portfolios.add(portfolio)

    # Long-short
    if ls_wgt is not None:
        pf_name = str(n_class) if labels is None else labels[-1]
        lposition = position[position[pf_col] == 0]
        sposition = position[position[pf_col] == n_class - 1]
        portfolio = make_long_short_portfolio(lposition, sposition, ls_wgt, rf, costfcn, keep_position, pf_name)
        portfolios.add(portfolio)

    log('Quantile portfolios created...')
    return portfolios


if __name__ == '__main__':
    os.chdir('../')


