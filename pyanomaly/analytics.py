import numpy as np
import pandas as pd
import statsmodels.api as sm

from pyanomaly.globals import *
from pyanomaly.numba_support import *
from pyanomaly.datatools import *
from pyanomaly.multiprocess import multiprocess

################################################################################
#
# CLASSIFICATION / GROUPING
#
################################################################################

def classify_array(array, split, ascending=True, by_array=None, labels=None):
    """Classify array into classes. if array contains nan, their classes are set to nan.

    Eg: If split=10, classify array into 10 equally-spaced groups (0, ..., 9); if split=[0.3, 0.7, 1.0], classify array
        into three groups (0, 1, 2) that correspond to 0.3, 03-0.7, 0.7-1.0 quantiles. When ascending=True, label 0
        indicates the smallest group.

    Args:
        array: Nx1 ndarray or Series to be grouped.
        split: number of classes or list of quantiles. (0.3, 0.7) is equivalent to (0.3, 0.7, 1.0).
        ascending: sorting order
        by_array: array based on which quantile values are determined. If None, by_array = array. Eg, array can be set
        to ME and by_array to NYSE-ME to group firms on size based on NYSE-quantiles.
        labels: list of class labels. If None, class labels are set to 0, ..., no. quantile-1.

    Returns:
        Nx1 ndarray or Series of classes. The output type follows the input type.
        Class labels are 0, ..., num_class-1.
    """
    # if np.isnan(array).any():
    #     raise ValueError('Classification failed. Data contains nan.')

    if isinstance(split, (int, np.integer)):  # number of classes
        quantiles = np.linspace(0, 1, split + 1)[1:]
    else:  # quantiles
        quantiles = split + ([1.0] if split[-1] < 1 else [])

    classes = np.zeros_like(array)

    if by_array is None:
        by_array = array
    by_array = by_array[~np.isnan(by_array)]

    if len(by_array) == 0:
        classes[:] = None
        return pd.Series(classes, index=array.index) if type(array) == pd.Series else classes

    values = np.quantile(by_array, quantiles[:-1])
    for i, v in enumerate(values[::-1]):
        classes[array <= v] = i + 1
    classes[np.isnan(array)] = np.nan

    if ascending:
        classes = len(quantiles) - 1 - classes

    if labels is not None:
        classes_ = np.empty_like(classes, dtype=type(labels[0]))
        for i in range(len(quantiles)):
            classes_[classes == i] = labels[i]
        classes = classes_

    return pd.Series(classes, index=array.index) if type(array) == pd.Series else classes


def classify_(data, col, split, ascending=True, by=None, labels=None):
    return classify_array(data[col], split, ascending, data[by] if by else None, labels)


def classify(data, col, split, ascending=True, by=None, labels=None):
    """ Classify data cross-sectionally. If data contains None, the class is set to None.

    Args:
        data: Dataframe with date/id index.
        col: column on which data is classified.
        split: number of classes or array of quantiles (see classify_array).
        ascending: sorting order
        by: column based on which quantile values are determined. If None, by = col

    Returns:
        pd.Series of class labels. Size = len(data), index = data.index.
    """
    cols = [col] + ([by] if by else [])

    data_ = data[cols]
    # classes = groupby_apply(data_.groupby(data.index.names[0]), classify_, col, split, ascending, by, labels)
    # return classes.sort_index(level=[1, 0])

    index = data_.index
    data_.reset_index(drop=True, inplace=True)
    classes = groupby_apply(data_.groupby(index.get_level_values(0)), classify_, col, split, ascending, by, labels)
    classes.sort_index(inplace=True)  # Restore the order.
    classes.index = index
    return classes


def weighted_mean(data, target_cols, weight_col, group_cols):
    """Calculate weighted mean of data[target_cols] for each group defined by group_cols.

    Args:
        data: Dataframe
        target_cols: (list of) column(s) to calculated weighted-mean
        weight_col: weight column name or Series or ndarray of weights.
        group_cols: (list of) grouping column(s)

    Returns:
        Dataframe of weighted mean with index=group_cols, columns=target_cols.

    """
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(weight_col, str):
        weight = data[weight_col]
    elif isinstance(weight_col, pd.Series) or isinstance(weight_col, np.ndarray):
        weight = weight_col
    else:
        raise ValueError(f'weight_col should be a column name, pd.Series or np.ndarray. {type(weight_col)} is given instead.')

    numer = data[target_cols].mul(weight, axis=0)
    denom = pd.notnull(data[target_cols]).mul(weight, axis=0)  # set weight=0 where target data is null.
    da = data[data.columns.intersection(group_cols)]  # data necessary for groupby (to save memory)
    gb1 = pd.concat([da, numer], axis=1).groupby(group_cols)
    gb2 = pd.concat([da, denom], axis=1).groupby(group_cols)
    retval = gb1[target_cols].sum().divide(gb2[target_cols].sum())
    return retval


################################################################################
#
# ONE-DIM / TWO-DIM SORT
#
################################################################################

def relabel_class(data, labels=None, axis=0, level=-1):
    """Relabel classes in columns/indexes of data.

    Args:
        data: Dataframe to be relabeled.
        labels: new class labels
        axis: 1: index, 2: column
        level: level of index/column.

    Returns:
        Relabeled data.
    """
    if labels is None:
        return data

    if axis == 0:
        name_mapping = {cls: labels[i] for i, cls in enumerate(data.index.get_level_values(level).unique())}
        return data.rename(index=name_mapping, level=level)
    else:
        name_mapping = {cls: labels[i] for i, cls in enumerate(data.columns.get_level_values(level).unique())}
        return data.rename(columns=name_mapping, level=level)


def append_long_short(data, label=None):
    """Add long-short to data. Long-short is defined as (first group - last group) in each date.

    Args:
        data: Dataframe with index[0] = date, index[-1] = class labels.
        label: long-short class label. If None, label=number of classes.

    Returns:
        data with long-short appended.
    """
    label = label or len(data.index.get_level_values(-1).unique())

    def append_long_short_(data):
        ls = data.iloc[0] - data.iloc[-1]
        ls.name = (*data.index[0][:-1], label)
        return data.append(ls)

    x = groupby_apply(data.groupby(level=0), append_long_short_)
    return x


def one_dim_sort(data, class_col, target_cols=None, weight_col=None, function='mean', add_long_short=False, labels=None):
    """One-dimensional sort. This function assumes that data has already been sorted/grouped and class labels given
    in class_col column. Aggregate data[target_cols] using data[class_col] and return aggregated results.

    Args:
        data: data to be grouped. Index must be date/id.
        class_col: class label column.
        target_cols: (list of) column(s) to aggregate. If None, target_cols = all numeric columns of data.
        weight_col: weight column. If given, weighted mean is returned.
        function: aggregate function. If it is other than 'mean', weight_col is ignored.
        add_long_short: add long-short result to the output.
        labels: list of class labels. If given, class labels are overwritten by this.

    Returns:
        Aggregated data. Index: date/class, columns: target_cols
    """
    date_col = data.index.names[0]
    if target_cols is None:  # target_cols = all columns in data except for non-numeric ones.
        target_cols = list(data.columns[(data.dtypes == float) | (data.dtypes == int)])
    else:
        if (not isinstance(target_cols, list)) and (not isinstance(target_cols, tuple)):  # single column
            target_cols = [target_cols]

    if (function == 'mean') and weight_col:  # value_weight
        ret_data = weighted_mean(data, target_cols, weight_col, [date_col, class_col])
    else:  # equal-weight
        gb = data.groupby([date_col, class_col])[target_cols]
        ret_data = eval(f'gb.{function}()')

    if add_long_short:
        ret_data = append_long_short(ret_data)

    ret_data = relabel_class(ret_data, labels, level=1)

    return ret_data


def two_dim_sort(data, class_col1, class_col2, target_cols=None, weight_col=None, function='mean',
                 add_long_short=False, labels1=None, labels2=None, output_dim=1):
    """Two-dimensional sort. This function assumes that data has already been sorted/grouped and class labels given
    in class_col1 and class_col2 columns. Aggregate data[target_cols] using data[class_col1, class_col2] and return
    aggregated results.

    Args:
        data: data to be grouped. Index must be date/id.
        class_col1(2): class label column for the 1st(2nd) dimension.
        target_cols: (list of) column(s) to aggregate. If None, target_cols = all numeric columns of data.
        weight_col: weight column. If given, weighted mean is returned.
        function: aggregate function. If it is other than 'mean', weight_col is ignored.
        add_long_short: add long-short result to the output.
        labels1(2): list of class labels for the 1st(2nd) dimension. If given, class labels are overwritten by this.
        output_dim: if 1, output is 1D DataFrame with index=date/class1/class2; if 2, output is 2D DataFrame with
        index=date/class1 and column=class2.

    Returns:
        Aggregated data.
        If output_dim = 1, Index: date/class1/class2, columns: target_cols
        If output_dim = 2 and len(target_cols) = 1, Index: date/class1, columns: class2
        If output_dim = 2 and len(target_cols) > 1, output is dict with keys = target_cols and values = 2D DataFrames
    """

    date_col = data.index.names[0]
    if target_cols is None:  # target_cols = all columns in data except for non-numeric ones.
        target_cols = list(data.columns[(data.dtypes == float) | (data.dtypes == int)])
    else:
        if (not isinstance(target_cols, list)) and (not isinstance(target_cols, tuple)):  # single column
            target_cols = [target_cols]

    if (function == 'mean') and weight_col:  # value_weight
        ret_data = weighted_mean(data, target_cols, weight_col, [date_col, class_col1, class_col2])
    else:  # equal-weight
        gb = data[[class_col1, class_col2] + target_cols].groupby([date_col, class_col1, class_col2])
        ret_data = eval(f'gb.{function}()')

    if add_long_short:
        ret_data = groupby_apply(ret_data.groupby(class_col1), append_long_short)
        ret_data = groupby_apply(ret_data.swaplevel().groupby(class_col2), append_long_short).swaplevel()
        ret_data = ret_data.sort_index()

    ret_data = relabel_class(ret_data, labels1, level=1)
    ret_data = relabel_class(ret_data, labels2, level=2)

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
        cov_type: see sm.OLS.fit(). eg, 'HAC' for Newey-West
        cov_kwds: see sm.OLS.fit(). eg, {'maxlags: 12} for cov_type = 'HAC'

    Returns:
        Series of t-values. Index: data.columns.
    """
    res = {}
    x = np.ones([len(data), 1])
    for col in data.columns:
        est = sm.OLS(data[col], x, missing='drop').fit(cov_type=cov_type, cov_kwds=cov_kwds)
        res[col] = est.tvalues[0]
    return pd.Series(res)


def time_series_average(data, cov_type='nonrobust', cov_kwds=None):
    """Calculate time-series mean and t-value of each column for each group.

    Args:
        data: DataFrame. If MultiIndex, the first index is assumed to be date index and data is grouped by the remaining
        index.
        cov_type: see t_value()
        cov_kwds: see t_value()

    Returns:
        Two DataFrames: mean and t-value.
        If data has multiple indexes, DataFrame with index: group labels, columns: data columns
        If data has only date index: DataFrame with index: data columns, columns: 'mean'/'t-stat'
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
        return mean.reindex(index), tval.reindex(index)  # reindex to keep the order of index


def rolling_beta(data, window, minobs=None, endo_col=None, exog_cols=None):
    """Run rolling OLS for each ID. This is faster than statsmodels' RollingOLS, but the output is limited to
    coefficients, R2, and idiosyncratic volatility.

    Args:
        data: Dataframe with index=date/id, sorted on id.
        window: window size.
        minobs: minimum number of observations in the window. If observations < min_n, result is nan.
        endo_col: y column.
        exog_cols: X columns (without constant). If endo_col = exog_cols = None, endo_col = data[:, 0],
        exog_cols = data[:, 1:].

    Returns:
        ndarray of list of coefficients, R2, and idiosyncratic volatility.
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
# CROSS-SECTIONAL REGRESSION
#
################################################################################

def crosssectional_regression(data, y_col, x_cols, add_constant=True, cov_type='nonrobust', cov_kwds=None):
    """Run cross-sectional OLS in each date and calculate the time-series means and t-values of the coefficients.

    Args:
        data: DataFrame with index=date/id
        y_col: y column
        x_cols: list of x columns
        add_constant: add constant to x
        cov_type: see t_value()
        cov_kwds: see t_value()

    Returns:
        time-series means, t-values of the coefficients, and time-series of coefficients.
        mean: Means of coefficients. Index: ('const') + x_cols, columns: 'mean'
        tval: t-values of coefficients. Index: ('const') + x_cols, columns: 't-stat'
        ceofs: Coefficient time-series. Index: dates, columns: ('const') + x_cols
    """
    data = data.dropna(subset=[y_col] + x_cols)

    # cross-sectional regression
    coefs = {}
    for k, v in data.groupby(level=0):
        y = v[y_col]
        x = sm.add_constant(v[x_cols]) if add_constant else v[x_cols]
        if x.shape[0] <= x.shape[1]:  # If not enough sample for regression, skip.
            continue
        model = sm.OLS(y, x).fit()

        coefs[k] = model.params
    coefs = pd.DataFrame(coefs).T

    # time series means and t-values of the cross-sectional regression coefficients
    mean, tval = time_series_average(coefs, cov_type, cov_kwds)

    return mean, tval, coefs


if __name__ == '__main__':
    os.chdir('../')


