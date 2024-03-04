"""This module defines analytic functions.

**Sorting**
    .. autosummary::
        :nosignatures:

        one_dim_sort
        two_dim_sort

    Auxiliary functions

    .. autosummary::
        :nosignatures:

        relabel_class
        weighted_mean
        append_long_short

**Time-Series Analysis**
    .. autosummary::
        :nosignatures:

        time_series_average
        grs_test

    Auxiliary functions

    .. autosummary::
        :nosignatures:

        t_stat

**Cross-sectional Analysis**
    .. autosummary::
        :nosignatures:

        crosssectional_regression

**Portfolio Analysis**
    .. autosummary::
        :nosignatures:

        make_position
        make_portfolio
        make_long_short_portfolio
        make_quantile_portfolios

    Auxiliary functions

    .. autosummary::
        :nosignatures:

        future_return

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

from pyanomaly.globals import *
from pyanomaly.numba_support import *
from pyanomaly.datatools import *
from pyanomaly.portfolio import Portfolio, Portfolios


################################################################################
#
# ONE-DIM / TWO-DIM SORT
#
################################################################################

def relabel_class(data, labels=None, axis=0, level=-1, col=None):
    """Relabel classes.

    Relabel (rename) columns, indexes, or column values of `data`. The existing labels (values) should be continuous
    integers starting from 0. The `data` is relabeled in-place.

    Args:
        data: DataFrame to be relabeled.
        labels (list): New class labels. Label 0 is replaced by the first element of `labels`, and so forth.
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


def weighted_mean(data, target_cols, weight_col, group_cols):
    """Calculate weighted means.

    Calculate weighted means of each column in `target_cols` within each group defined by `group_cols`.

    Args:
        data: DataFrame.
        target_cols: (List of) column(s) to calculate weighted-mean.
        weight_col: Weight column name or Series or ndarray of weights.
        group_cols: (List of) grouping column(s).

    Returns:
        DataFrame of weighted means with index = `group_cols` and columns = `target_cols`.

    Examples:
        If `data` is a panel with index = date/permno and column 'ret' contains returns
        and 'me' contains market equity at t-1, value-weighted returns can be obtained as follows:

        >>> wmean = weighted_mean(data, 'ret', 'me', 'date')
    """

    target_cols = to_list(target_cols)
    group_cols = to_list(group_cols)
    if is_iterable(weight_col):
        weight = weight_col
    else:
        weight = data[weight_col]

    weight = weight.to_numpy().reshape([-1, 1])
    target = data.loc[:, target_cols].to_numpy()
    denom = np.multiply(~np.isnan(target), weight)  # set weight=0 where target data is null.
    numer = np.multiply(target, weight)

    da = data.loc[:, data.columns.intersection(group_cols)]  # data necessary for groupby (to save memory)
    for i, col in enumerate(target_cols):
        da[col] = numer[:, i]
        da[col + '_w'] = denom[:, i]
    weight_cols = [col + '_w' for col in target_cols]

    gb = da.groupby(group_cols)
    return gb[target_cols].sum() / gb[weight_cols].sum().to_numpy()


def append_long_short(data, level=-1, l_label=None, s_label=None, ls_label=None):
    """Add long-short to a quantile data.

    Long-short is defined as (first group - last group) in each date.
    If labels are not given, long-short will be (class 0 - class N-1), where N is the number of classes,
    and the class label of the long-short is set to N.

    Args:
        data: DataFrame with index = date/class1/class2/....
        level: Index level to make long-short on. Default to the last level.
        l_label: Label of the long class. If None, `l_label` = 0.
        s_label: Label of the short class. If None, `s_label` = N-1.
        ls_label: Label of the long-short. If None, `ls_label` = N.

    Returns:
        The `data` with long-short appended.
    """

    if ((not l_label) or (not s_label)) and (not is_numeric(data.index.get_level_values(-1))):
        raise ValueError('The l_label and s_label should be provided for non-numeric class labels.')

    data2 = data.swaplevel(-1, level).unstack()
    classes = list(data2.columns.get_level_values(-1).unique())
    l_label = l_label or 0
    s_label = s_label or np.max(classes)
    ls_label = ls_label or np.max(classes) + 1

    for col in data:
        if isinstance(col, tuple):
            data2[(*col, ls_label)] = data2[(*col, l_label)] - data2[(*col, s_label)]
        else:
            data2[(col, ls_label)] = data2[(col, l_label)] - data2[(col, s_label)]

    try:  # Pandas 2.0 or later
        return data2.stack(future_stack=True).dropna().swaplevel(-1, level)[data.columns]
    except:
        return data2.stack(dropna=True).swaplevel(-1, level)[data.columns]


def one_dim_sort(data, class_col, target_cols=None, weight_col=None, function='mean', add_long_short=True):
    """One-dimensional sort.

    This function assumes that `data` has already been sorted/grouped and class labels are given
    in `class_col` column. Aggregate `target_cols` values using `class_col` and return aggregated results.
    Class labels in `class_col` should be 0, 1, ...

    Args:
        data: DataFrame to be grouped. Index must be date/id.
        class_col: Class label column.
        target_cols: (List of) column(s) to aggregate. If None, `target_cols` = all numeric columns of `data`.
        weight_col: Weight column. If given, weighted mean is returned. Applicable only when `function` = 'mean'.
        function: Aggregate function, e.g., 'sum', 'mean', 'count', or a list of functions.
        add_long_short (bool): Add long-short to the output.

    Returns:
        Aggregated data with index = date/class, columns = `target_cols`. If `function` is a list of functions,
        the columns has two levels: first level = `target_cols` and second level = `function`.
    """

    date_col = data.index.names[0]
    if target_cols is None:  # target_cols = all columns in data except for non-numeric ones.
        target_cols = [col for col in data.columns if is_numeric(data[col])]
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

    if add_long_short:
        ret_data = append_long_short(ret_data)

    return ret_data


def two_dim_sort(data, class_col1, class_col2, target_cols=None, weight_col=None, function='mean',
                 add_long_short=True, output_dim=1):
    """Two-dimensional sort.

    This function assumes that `data` has already been sorted/grouped and class labels are given
    in `class_col1` and `class_col2` columns. Aggregate `target_cols` values using `class_col1` and `class_col2` and
    return aggregated results. Class labels in `class_col1(2)` should be 0, 1, ...

    Args:
        data: Data to be grouped. Index must be date/id.
        class_col1: Class label column for the 1st dimension.
        class_col2: Class label column for the 2nd dimension.
        target_cols: (List of) column(s) to aggregate. If None, `target_cols` = all numeric columns of `data`.
        weight_col: Weight column. If given, weighted mean is returned. Applicable only when `function` = 'mean'.
        function: Aggregate function, e.g., 'sum', 'mean', 'count'.
        add_long_short (bool): Add long-short to the output.
        output_dim: If 1, output is a DataFrame with index = date/class1/class2; if 2, output is a DataFrame with
            index = date/class1 and column = class2.

    Returns:
        Aggregated data (DataFrame or dict of DataFrames).

        * If `output_dim` = 1, index = date/class1/class2 and columns = `target_cols`.
        * If `output_dim` = 2 and `len(target_cols)` = 1, index = date/class1 and columns = class2.
        * If `output_dim` = 2 and `len(target_cols)` > 1, output is dict with keys = `target_cols` and
          values = DataFrames (as in the second case).
    """

    date_col = data.index.names[0]
    if target_cols is None:  # target_cols = all columns in data except for non-numeric ones.
        target_cols = [col for col in data.columns if is_numeric(data[col])]
        # target_cols = list(data.columns[is_numeric(data.dtypes)])
    else:
        target_cols = to_list(target_cols)

    if (function == 'mean') and weight_col:  # value_weight
        ret_data = weighted_mean(data, target_cols, weight_col, [date_col, class_col1, class_col2])
    else:  # equal-weight
        gb = data.groupby([date_col, class_col1, class_col2])[target_cols]
        ret_data = eval(f'gb.{function}()')

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

def t_stat(data, cov_type='nonrobust', cov_kwds=None):
    """Calculate `t`-statistic.

    Calculate `t`-statistic for each column of `data` under H0: x = 0.

    Args:
        data: Series, DataFrame, or ndarray with each column containing samples.
        cov_type: Covariance estimator: e.g., 'HAC' for Newey-West.
        cov_kwds: Parameters required for the chosen covariance estimator: e.g., {'maxlags: 12} for `cov_type` = 'HAC'.

    Returns:
        `t`-stat. Float (if `data` is one dimensional) or Series with index = `data.columns`.

    Note:
        See `statsmodels.api.OLS.fit`_ for possible values of `cov_type` and `cov_kwds`.

        .. _statsmodels.api.OLS.fit: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html.
    """

    x = np.ones([len(data), 1])
    if isinstance(data, pd.Series) or isinstance(data, np.ndarray):
        est = sm.OLS(data, x, missing='drop').fit(cov_type=cov_type, cov_kwds=cov_kwds)
        return est.tvalues.iloc[0]
    else:  # DataFrame
        res = {}
        for col in data.columns:
            est = sm.OLS(data[col], x, missing='drop').fit(cov_type=cov_type, cov_kwds=cov_kwds)
            res[col] = est.tvalues.iloc[0]
        res = pd.Series(res)
        res.index = res.index.set_names(data.columns.names)
        return res


def time_series_average(data, cov_type='nonrobust', cov_kwds=None):
    """Calculate time-series mean and `t`-statistic.

    Time-series mean and `t`-statistic are calculated for each column of  `data`. The `data` can be either a
    time-series data (index = date) or a panel data (index = date/id). If it is a panel, time-series mean and
    t-statistic are calculated for each id.

    Args:
        data: DataFrame. Data to analyze. If MultiIndex, the first index must be date.
        cov_type: Covariance estimator. See :func:`t_stat`.
        cov_kwds: Parameters required for the chosen covariance estimator. See :func:`t_stat`.

    Returns:
        * mean (DataFrame).
        * `t`-stat (DataFrame).

        If `data` has MultiIndex, mean (`t`-stat) has index = `data.index[1:]` and columns = `data.columns`.
        Otherwise, mean (`t`-stat) has index = `data.columns` and columns = 'mean' ('t-stat').
    """

    if not isinstance(data.index, pd.MultiIndex):  # single (date) index
        mean = data.mean().to_frame(name='mean')
        tval = t_stat(data, cov_type, cov_kwds).to_frame(name='t-stat')
        return mean, tval
    else:  # MultiIndex.
        data = data.reset_index(level=0, drop=True)
        index = data.index.unique()
        if is_numeric(index):  # unique() can change the order when there are missing classes.
            index = index.sort_values()  # => Sort the classes.

        mean = data.groupby(index.names).mean()
        tval = data.groupby(index.names).apply(t_stat, cov_type, cov_kwds)
        if isinstance(data, pd.DataFrame):
            tval.columns.names = mean.columns.names
        return mean.reindex(index), tval.reindex(index)  # reindex to keep the order of index


def grs_test(assets, factors):
    """Run GRS (Gibbons, Ross, and Shanken, 1989) test.

    Args:
        assets: T x N DataFrame or ndarray of asset returns.
        factors: T x F DataFrame or ndarray of factor returns.

    Returns:
        * pricing error (alpha.T * inv(Sigma) * alpha)
        * squared Sharpe ratio of the factors
        * GRS statistic
        * p value
    """

    T, N = assets.shape
    F = factors.shape[1]

    X = factors.values
    X = sm.add_constant(X)

    alpha = np.full(assets.shape[1], np.nan)
    e = np.full(assets.shape, np.nan)
    for i, asset in enumerate(assets):
        y = assets[asset]
        beta = (np.linalg.solve(X.T @ X, X.T @ y))
        y_pre = X.dot(beta)
        alpha[i] = beta[0]
        e[:, i] = y - y_pre

    V_e = np.cov(e.T, ddof=F + 1)
    mu_f = np.mean(factors, axis=0)
    V_f = np.cov(factors.T)

    ir2 = alpha.T @ np.linalg.solve(V_e, alpha)
    sr2 = mu_f.T @ np.linalg.solve(V_f, mu_f)
    grs = T / N * (T - N - F) / (T - F - 1) * ir2 / (1 + sr2)
    p_val = 1 - f.cdf(grs, N, T - N - F)

    return ir2, sr2, grs, p_val


################################################################################
#
# CROSS-SECTIONAL ANALYSIS
#
################################################################################

def crosssectional_regression(data, endo_col, exog_cols, add_constant=True, cov_type='nonrobust', cov_kwds=None):
    """Run cross-sectional OLS.

     Run cross-sectional OLS on each date and calculate the time-series means and t-stats of the coefficients.

    Args:
        data: DataFrame with index = date/id.
        endo_col: y column.
        exog_cols: List of X columns.
        add_constant (bool): Add constant to X.
        cov_type: Covariance estimator. See :func:`t_stat`.
        cov_kwds: Parameters required for the chosen covariance estimator. See :func:`t_stat`.

    Returns:
        * mean (DataFrame). Time-series means of coefficients with index = ('const' +) `exog_cols` and columns = 'mean'.
        * tval (DataFrame). `t`-statistics of coefficients with index = ('const' +) `exog_cols` and columns = 't-stat'.
        * coefs (DataFrame). Coefficient time-series with index = dates and columns = ('const' +) `exog_cols`.
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


def future_return(ret, period=1):
    """Compute future returns.

    Compute `period`-period ahead returns. If `ret` has a MultiIndex of date/id, future returns are calculated for each
    id.

    Args:
        ret: Series of returns with index = date or date/id. If index = date/id, `ret` must be sorted on id/date.
        period: Target period.

    Returns:
        Series of future returns.
    """

    if period == 1:
        futret = ret.shift(-1)
    else:
        futret = np.exp(np.log(ret + 1).shift(-period).rolling(period).sum()) - 1

    if isinstance(ret.index, pd.MultiIndex):
        ids = ret.index.get_level_values(1)
        futret[ids != np.roll(ids, -period)] = np.nan

    return futret


def make_position(data, ret_col, weight_col=None, pf_col=None, other_cols=None):
    """Make portfolio position data.

    To construct and evaluate a portfolio using :class:`~.portfolio.Portfolio`, position data is required.
    This function makes the position data from `data`, which is a panel of securities.
    The position data is generated via the following operations:

        - Change column names as assumed in :class:`~.portfolio.Portfolio`:

            - 'date': Date column.
            - 'id': Security id column.
            - 'ret': Return column.
            - 'wgt': Weight column.
        - Normalize weights so that their cross-sectional sum becomes 1 within each portfolio.

    Args:
        data: DataFrame with index = date/id.
        ret_col: Return column of `data`. Return should be over t to t+1.
        weight_col: Weight column of `data`. If None, constituents are equally weighted.
        pf_col: Portfolio column of `data`, i.e., a column that maps securities with portfolios.
            This can be None if the input data is for one portfolio.
        other_cols: Other columns of `data` to include in the ``position`` attribute of ``Portfolio``.

    Returns:
        Position DataFrame with index = 'date' and columns = ['id', 'ret', 'wgt'] + `other_cols`.
    """

    columns = unique_list([ret_col, weight_col, pf_col, other_cols])
    position = data[columns].copy()

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

    position = position.dropna(subset=['ret', 'wgt'])  # Drop nan returns.

    if pf_col:  # Normalize weights per date/portfolio.
        position['wgt'] = position.wgt / position.groupby(['date', pf_col]).wgt.transform('sum')
    else:  # Normalize weights per date.
        position['wgt'] = position.wgt / position.groupby('date').wgt.transform('sum')

    return position.reset_index('id').sort_index()


def make_portfolio(data, ret_col, weight_col=None, rf=None, costfcn=None, keep_position=True, name=''):
    """Make a portfolio.

    This function creates portfolio position data from `data` and construct a portfolio from it.

    Args:
        data: DataFrame with index = date/id.
        ret_col: Return column of `data`. Return should be over t to t+1.
        weight_col: Weight column of `data`. If None, constituents are equally weighted.
        rf: Series or DataFrame of risk-free rates. The index should be date.
        costfcn: Transaction cost. See :meth:`Portfolio.costfcn() <pyanomaly.portfolio.Portfolio.costfcn>`.
        keep_position: If True, keep the position information in the returned value. If position information is not
            needed, set this to False to save memory.
        name: Portfolio name.

    Returns:
        :class:`~pyanomaly.portfolio.Portfolio` object.
    """

    position = make_position(data, ret_col, weight_col)
    return Portfolio(name, position, rf=rf, costfcn=costfcn, keep_position=keep_position)


def make_long_short_portfolio(lposition, sposition, rf=None, costfcn=None, keep_position=True,
                              name='H-L', ls_wgt=(1, -1)):
    """Make a long-short portfolio.

    Args:
        lposition: DataFrame. Long position data. See :func:`make_position` for the data format.
        sposition: DataFrame. Short position data.
        rf: Series or DataFrame of risk-free rates. The index should be date.
        costfcn: Transaction cost. See :meth:`Portfolio.costfcn() <pyanomaly.portfolio.Portfolio.costfcn>`.
        keep_position: If True, keep the position information in the returned value. If position information is not
            needed, set this to False to save memory.
        name: Portfolio name.
        ls_wgt: Long-short weights. (1, -1) means 1:1 long-short.

    Returns:
        :class:`~.portfolio.Portfolio` object.
    """

    # Copy data as their values ('wgt') are changed.
    lposition = lposition.copy()
    sposition = sposition.copy()

    lposition['wgt'] *= ls_wgt[0]
    sposition['wgt'] *= ls_wgt[1]
    position = pd.concat([lposition, sposition]).sort_index()
    portfolio = Portfolio(name, position, rf=rf, costfcn=costfcn, keep_position=keep_position)
    return portfolio


def make_quantile_portfolios(data, q_col, ret_col, weight_col=None, rf=None, costfcn=None,
                             keep_position=True, names=None, ls_wgt=(1, -1)):
    """Make quantile portfolios.

    This function makes quantile portfolios and the long-short portfolio from `data`.

    Args:
        data: DataFrame with index = date/id.
        q_col: Column of `data` that maps a security with quantiles (portfolios). The values should be integers
            starting from 0.
        ret_col: Return column of `data`. Return should be over t to t+1.
        weight_col: Weight column of `data`. If None, constituents are equally weighted.
        rf: Series or DataFrame of risk-free rates. The index should be date.
        costfcn: Transaction cost. See :meth:`Portfolio.costfcn() <pyanomaly.portfolio.Portfolio.costfcn>`.
        keep_position: If True, keep the position information in the returned value. If position information is not
            needed, set this to False to save memory.
        names: Portfolio names. If None, the values in `pf_col` are used.
        ls_wgt: Long-short weights. (1, -1) means 1:1 long-short. If None, long-short portfolio is not constructed.

    Returns:
        :class:`~.portfolio.Portfolios` object.
    """

    log(f'Making quantile portfolios for {q_col}...')

    position = make_position(data, ret_col, weight_col, q_col)
    n_pf = int(position[q_col].max()) + 1

    portfolios = Portfolios()
    for k in range(n_pf):
        pf_name = str(k) if names is None else names[k]
        position_ = position.loc[position[q_col] == k, position.columns.difference([q_col])]
        portfolios.add(Portfolio(pf_name, position_, rf=rf, costfcn=costfcn, keep_position=keep_position))

        if k == 0:
            lposition = position_
        elif k == n_pf - 1:
            sposition = position_

    # Long-short
    if ls_wgt is not None:
        pf_name = str(n_pf) if names is None else names[-1]
        portfolios.add(make_long_short_portfolio(lposition, sposition, rf, costfcn, keep_position, pf_name, ls_wgt))

    log('Quantile portfolios created...')
    return portfolios


# def get_cumulative_return(ret, period=1, lag=0):
#
#     cumret = roll_cumret(ret.to_numpy(), period, lag)
#
#     if isinstance(ret.index, pd.MultiIndex):
#         ids = ret.index.get_level_values(-1).values
#         cumret[(ids != np.roll(ids, period)).astype(bool)] = np.nan
#     return cumret
#
#
# def get_cumulative_return2(ret, period=1, lag=0):
#     if isinstance(ret.index, pd.MultiIndex):
#         gsize = ret.groupby(level=1).size().to_numpy()
#         return apply_to_groups_jit(ret.to_numpy(), gsize, roll_cumret, None, period, lag)
#
#
# # @njit
# def roll_cumret(ret, period=1, lag=0):
#     """Compute cumulative returns between t-`period` and t-`lag`.
#
#     When `ret` is monthly returns, 12-month momentum can be obtained by setting `period` = 12 and `lag` = 1.
#     A negative `period` will generate future returns: eg, `period` = -1 and `lag` = 0 for one-period ahead
#     return; `period` = -3 and `lag` = -1 for two-period ahead
#     return starting from t+1.
#
#     Args:
#         ret: Series of returns in `base_freq` or a return column of the `data` attribute.
#         period: Target horizon in `base_freq`. (+) for past returns and (-) for future returns.
#         lag: Period (in `base_freq`) to calculate returns from.
#
#     Returns:
#         Series of cumulative returns.
#
#     NOTE:
#         Returns, `ret`, should be in `base_freq`. If `base_freq` = ANNUAL, `ret` should be annual
#         returns regardless of the value of `freq`.
#     """
#
#     # if not logscale:
#     #     cumret = np.exp(cumret) - 1
#
#     if period == 1:
#         return ret
#     elif period == -1:
#         return shift(ret, -1)
#     elif period > 0:
#         if lag:
#             ret = shift(ret, lag)
#         return np.exp(roll_sum2(np.log(ret + 1), period - lag, period - lag)) - 1
#     elif period < 0:
#         return np.exp(roll_sum2(np.log(shift(ret, period) + 1), -(period - lag), -(period - lag))) - 1
#     else:
#         raise ValueError('x')

if __name__ == '__main__':
    os.chdir('../')


