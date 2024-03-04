"""This module defines classes for panel data analysis.

    .. autosummary::
        :nosignatures:

        Panel
        FCPanel
"""

import copy
import numpy as np
import pandas as pd
import json
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from pyanomaly.globals import *
from pyanomaly.datatools import *
from pyanomaly.fileio import write_to_file, read_from_file


class Panel:
    """Base class for panel data analysis.

    This class stores a panel data, `data`, and offers various tools to handle and analyze it.
    The `data` should be a Pandas DataFrame with a MultiIndex = date/id, i.e., the first  index should be a time-series
    identifier (Pandas datetime type) and the second index a cross-section identifier. It should be sorted on id/date.

    Args:
        data: Panel data DataFrame with index = date/id, sorted on id/date.
        freq: Frequency of `data`. ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        base_freq: Frequency of `data` values. For example, if `data` has annual values populated monthly,
            `freq` = MONTHLY and `base_freq` = ANNUAL. If None, `base_freq` = `freq`.

    **Attributes**

    Attributes:
        data: DataFrame with index = date/id, sorted on id/date that stores a panel data.
            The items of `data` can be accessed by ``Panel.__get_item__()``: For a ``Panel`` instance ``panel``,
            ``panel[cols]`` and ``panel[rows, cols]`` are equivalent to ``panel.data[col]`` and
            ``panel.data.loc[rows, cols]``, respectively.
        freq: Frequency of ``data``. ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        base_freq: Frequency of ``data`` values. ANNUAL, QUARTERLY, MONTHLY, or DAILY.

    **Methods**

    .. autosummary::
        :nosignatures:

        is_sorted
        id_idx
        date_idx
        id_values
        date_values
        get_id_group
        get_date_group
        get_id_group_size
        get_date_group_size
        get_id_group_index
        get_date_group_index
        apply_to_ids
        apply_to_dates
        populate
        merge
        inspect_data
        filter
        get_row_count
        remove_rows
        shift
        diff
        pct_change
        cumret
        futret
        rolling
        rolling_regression
        copy
        copy_from
        save
        load
        clean_memory
    """

    def __init__(self, data=None, freq=MONTHLY, base_freq=None):
        if data is not None:
            assert self.is_sorted(data)

        self.data = data
        self.freq = freq
        self.base_freq = base_freq or freq

        self._gb_id_idx = None
        self._gb_id_size = None  # Ndarray of ID group sizes

        self._gb_date_idx = None
        self._gb_date_size = None

        self._row_count = None
        self._row_count1 = None

    def __getitem__(self, idx):
        """
            ``panel[cols]``: Same as ``panel.data[cols]``.
            ``panel[rows, cols]``: Same as ``panel.data.loc[rows, cols]``.
        """

        if isinstance(idx, tuple):
            return self.data.loc[idx[0], idx[1]]
        else:
            return self.data.loc[:, idx]

    def __setitem__(self, idx, val):
        """
            ``panel[cols]``: Same as ``panel.data[cols]``.
            ``panel[rows, cols]``: Same as ``panel.data.loc[rows, cols]``.
        """

        if isinstance(idx, tuple):
            self.data.loc[idx[0], idx[1]] = val
        else:
            self.data[idx] = val

    def is_sorted(self, data=None):
        """Check if a panel data is sorted on id/date.

        Args:
            data: Panel DataFrame with index = date/id. If None, `data` = ``data``.

        Returns:
            Bool. True if `data` is sorted on id/date.
        """

        if data is None:
            return self.data.index.swaplevel().is_monotonic_increasing
        else:
            return data.index.swaplevel().is_monotonic_increasing

    def id_idx(self):
        """Get id index name.

        Returns:
            Id index name.
        """

        return self.data.index.names[-1]

    def date_idx(self):
        """Get date index name.

        Returns:
            Date index name.
        """

        return self.data.index.names[0]

    def id_values(self):
        """Get id index values.

        Returns:
            Id Index.
        """

        return self.data.index.get_level_values(-1)

    def date_values(self):
        """Get date index values.

        Returns:
            Date Index.
        """

        return self.data.index.get_level_values(0)

    def get_id_group(self):
        """Get id group.

        Same as ``Panel.data.groupby(level=1)``.

        Returns:
            Pandas GroupBy object.
        """
        # if self._gb_id is not None:
        #     if self._gb_id_size is None:
        #         self._gb_id_size = self._gb_id.size().to_numpy()
        #
        #     if np.sum(self._gb_id_size) == self.data.shape[0]:
        #         return self._gb_id
        #
        # self._gb_id = self.data.groupby(level=1)
        # return self._gb_id
        return self.data.groupby(level=1)

    def get_date_group(self):
        """Get date group.

        Same as ``Panel.data.groupby(level=0)``.

        Returns:
            Pandas GroupBy object.
        """

        return self.data.groupby(level=0)

    def get_id_group_size(self):
        """Get id group sizes.

        Returns:
            Ndarray of id group sizes.
        """

        if (self._gb_id_size is None) or (np.sum(self._gb_id_size) != self.data.shape[0]):
            _gb_id = self.get_id_group()
            self._gb_id_idx = list(_gb_id.indices.values())
            self._gb_id_size = _gb_id.size().to_numpy()

        return self._gb_id_size

    def get_date_group_size(self):
        """Get date group sizes.

        Returns:
            Ndarray of date group sizes.
        """

        if (self._gb_date_size is None) or (np.sum(self._gb_date_size) != self.data.shape[0]):
            _gb_date = self.get_date_group()
            self._gb_date_idx = list(_gb_date.indices.values())
            self._gb_date_size = _gb_date.size().to_numpy()

        return self._gb_date_size

    def get_id_group_index(self):
        """Get id group indices.

        Returns:
            List of id group indices.
        """

        if (self._gb_id_size is None) or (np.sum(self._gb_id_size) != self.data.shape[0]):
            _gb_id = self.get_id_group()
            self._gb_id_idx = list(_gb_id.indices.values())
            self._gb_id_size = _gb_id.size().to_numpy()

        return self._gb_id_idx

    def get_date_group_index(self):
        """Get date group indices.

        Returns:
            List of date group indices.
        """

        if (self._gb_date_size is None) or (np.sum(self._gb_date_size) != self.data.shape[0]):
            _gb_date = self.get_date_group()
            self._gb_date_idx = list(_gb_date.indices.values())
            self._gb_date_size = _gb_date.size().to_numpy()

        return self._gb_date_idx

    def _to_value(self, data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_numpy()
        elif isinstance(data, (list, str)):
            return self[data].to_numpy()
        else:  # ndarray
            return data

    def _to_frame(self, value, data):
        if isinstance(data, pd.Series):
            return pd.Series(value, index=data.index, name=data.name)
        elif isinstance(data, pd.DataFrame):
            return pd.DataFrame(value, index=data.index, columns=data.columns)
        elif isinstance(data, str):
            return pd.Series(value, index=self.data.index, name=data)
        elif isinstance(data, list):
            return pd.DataFrame(value, index=self.data.index, columns=data)
        else:  # ndarray
            return value

    @staticmethod
    @njit(cache=True)
    def _get_sumsample(data, cum_gsize, i, e, m):
        ridx_ = np.arange(i, e, m)
        gsize_ = np.zeros_like(cum_gsize)

        k = 0
        for j in ridx_:
            if j >= cum_gsize[k]:
                while j >= cum_gsize[k]:
                    k += 1

            gsize_[k] += 1

        data_ = data[ridx_]
        gsize_ = gsize_[gsize_ > 0]
        return data_, gsize_, ridx_

    def apply_to_ids(self, data, function, n_ret, *args, data2=None, looper=None):
        """Apply a function to each id group.

        This method groups `data` by id and applies `function` to each id group.

        Args:
            data: DataFrame, Series, ndarray, or (list of) columns. It should have the same length and order as the
                  ``data`` attribute.
            function: Function to apply to groups. Its arguments should be (`gbdata`, `*args`) or
                (`gbdata`, `gbdata2`, `*args`), where `gbdata` (`gbdata2`) is a group of `data` (`data2`).
            n_ret: Number of returns of `function`. If None, it is assumed to be the same as the column size of `data`.
                If `looper` is ``apply_to_groups``, this argument is ignored.
            *args: Additional arguments of `function`.
            data2: DataFrame, Series, ndarray, str, or int. Optional argument when `function`
                requires two sets of input data.
            looper: Looping function: :func:`~.datatools.apply_to_groups`, :func:`~.datatools.apply_to_groups_jit`, or
                :func:`~.datatools.apply_to_groups_reduce_jit`. If `looper` is None, ``apply_to_groups_jit`` is used if
                `function` is jitted, otherwise, ``apply_to_groups`` is used.

        Returns:
            Concatenated value of the outputs of `function`.

        See Also:
            :func:`~.datatools.apply_to_groups`
            :func:`~.datatools.apply_to_groups_jit`
            :func:`~.datatools.apply_to_groups_reduce_jit`
        """

        if not looper:
            if is_jitted(function):
                looper = apply_to_groups_jit
            else:
                looper = apply_to_groups

        value = self._to_value(data)
        value2 = self._to_value(data2) if data2 is not None else None
        gsize = self.get_id_group_size()

        m = int(self.freq / self.base_freq)

        if m == 1:
            if looper == apply_to_groups:
                return looper(value, gsize, function, *args, data2=value2)
            else:
                return looper(value, gsize, function, n_ret, *args, data2=value2)
        else:
            cum_gsize = gsize.cumsum()
            e = len(value)
            retval = []
            ridx = []
            for i in range(m):
                value_, gsize_, ridx_ = self._get_sumsample(value, cum_gsize, i, e, m)
                value2_ = value2[ridx_] if value2 is not None else None
                ridx.append(ridx_)
                retval.append(looper(value_, gsize_, function, n_ret, *args, data2=value2_))

            return np.concatenate(retval)[np.argsort(np.concatenate(ridx))]

    def apply_to_dates(self, data, function, n_ret, *args, data2=None, looper=None):
        """Apply a function to each date group.

        This method groups `data` by date and applies `function` to each date group.

        Args:
            data: DataFrame, Series, ndarray, or (list of) columns. It should have the same length and order as the
                  ``data`` attribute.
            function: Function to apply to groups. Its arguments should be (`gbdata`, `*args`) or
                (`gbdata`, `gbdata2`, `*args`), where `gbdata` (`gbdata2`) is a group of `data` (`data2`).
            n_ret: Number of returns of `function`. If None, it is assumed to be the same as the column size of `data`.
                If `looper` is ``apply_to_groups``, this argument is ignored.
            *args: Additional arguments of `function`.
            data2: DataFrame, Series, ndarray, str, or int. Optional argument when `function`
                requires two sets of input data.
            looper: Looping function: :func:`~.datatools.apply_to_groups`, :func:`~.datatools.apply_to_groups_jit`, or
                :func:`~.datatools.apply_to_groups_reduce_jit`. If `looper` is None, ``apply_to_groups_jit`` is used if
                `function` is jitted, otherwise, ``apply_to_groups`` is used.

        Returns:
            Concatenated value of the outputs of `function`.

        See Also:
            :func:`~.datatools.apply_to_groups`
            :func:`~.datatools.apply_to_groups_jit`
            :func:`~.datatools.apply_to_groups_reduce_jit`
        """

        if not looper:
            if is_jitted(function):
                looper = apply_to_groups_jit
            else:
                looper = apply_to_groups

        value = self._to_value(data)
        value2 = self._to_value(data2) if data2 is not None else None
        ginfo = self.get_date_group_index()

        if looper == apply_to_groups:
            return looper(value, ginfo, function, *args, data2=value2)
        else:
            return looper(value, ginfo, function, n_ret, *args, data2=value2)

    def populate(self, freq=None, method='ffill', limit=None, lag=0, new_date_col=None):
        """Populate data.

        Populate data to `freq` frequency and shift the populated data by `lag` period(s).

        Args:
            freq: Frequency to populate: ANNUAL, QUARTERLY, MONTHLY, or DAILY. If None, `freq` is set to the data
                frequency (``freq``), and missing dates are added.
            method: Filling method for newly added rows. 'ffill': forward fill, None: nan.
            limit: Maximum number of rows to forward-fill.
            lag: Minimum periods between new date and original date. If `freq` = MONTHLY, `lag` = 4 shifts data by
                4 months, which means data are available at least 4 months later.
            new_date_idx: Name of the new (populated) date index. If None, use the current date index name. If given,
                the original date index is kept as a column.

        See Also:
            :func:`.datatools.populate`
        """

        freq = freq or self.freq
        elapsed_time(f'Populating data...')

        self.data = populate(self.data, freq, method, limit, new_date_col)
        self.freq = freq

        if lag:  # Shift data by 'lag' months.
            self.data = self.get_id_group().shift(lag)

        elapsed_time(f'Data populated.')

    def merge(self, right, on=None, right_on=None, how='left', drop_duplicates='right', suffixes=None, method=None):
        """Merge with another data.

        Merge the ``data`` attribute with `right`.

        Args:
            right: Panel, Series, or DataFrame to merge with.
            on: (List of) column(s) to merge on. If None, merge on index.
            right_on: (List of) column(s) of `right` to merge on. If None, `right_on` = `on`.
            how: Merge method: 'inner', 'outer', 'left', or 'right'.
            drop_duplicates: how to handle duplicate columns. 'left': keep right, 'right': keep left,
                None: keep both. If None, `suffixes` should be provided.
            suffixes: A tuple of suffixes for duplicate columns, e.g., suffixes=('_x', '_y') will add '_x' and '_y'
                to the left and right duplicate columns, respectively.
            method: None or 'pandas'. None uses an internal merge algorithm for left-merge; 'pandas' uses ``pd.merge()``
                internally. If `how` is not 'left', this option is ignored and ``pd.merge()`` is always used.

        See Also:
            :func:`.datatools.merge`
        """

        if isinstance(right, Panel):
            right = right.data

        self.data = merge(self.data, right, on, right_on, how, drop_duplicates, suffixes, method)

    def inspect_data(self, columns=None, option=['summary']):
        """Inspect data.

        See :func:`.datatools.inspect_data`.

        Args:
            columns: List of columns to inspect.
            option: List of items to display.
        """

        if not columns:
            inspect_data(self.data, option)
        else:
            inspect_data(self.data.loc[:, columns], option)

    def _filter_data(self, filter, keep_row=False):
        column = filter[0]
        condition = filter[1]
        value = filter[2]

        log(f"filter: {column} {condition} {value}")
        if condition == '==':
            if keep_row:
                self.data.loc[self.data[column] != value, :] = None
            else:
                self.data = self.data[self.data[column] == value]
        elif condition == '!=':
            if keep_row:
                self.data.loc[self.data[column] == value, :] = None
            else:
                self.data = self.data[self.data[column] != value]
        elif condition == '>':
            if keep_row:
                self.data.loc[self.data[column] <= value, :] = None
            else:
                self.data = self.data[self.data[column] > value]
        elif condition == '<':
            if keep_row:
                self.data.loc[self.data[column] >= value, :] = None
            else:
                self.data = self.data[self.data[column] < value]
        elif condition == '>=':
            if keep_row:
                self.data.loc[self.data[column] < value, :] = None
            else:
                self.data = self.data[self.data[column] >= value]
        elif condition == '<=':
            if keep_row:
                self.data.loc[self.data[column] > value, :] = None
            else:
                self.data = self.data[self.data[column] <= value]
        elif condition == 'in':
            if keep_row:
                self.data.loc[~self.data[column].isin(value), :] = None
            else:
                self.data = self.data[self.data[column].isin(value)]
        elif condition == 'not in':
            if keep_row:
                self.data.loc[self.data[column].isin(value), :] = None
            else:
                self.data = self.data[~self.data[column].isin(value)]
        else:
            raise ValueError(f'Unrecognized filtering condition: {condition}')

    def filter(self, filters, keep_row=False):
        """Filter data.

        Filter the panel data  using `filters`. A filter is a tuple of three elements:

            * filter[0]: column to apply the filter to.
            * filter[1]: filter condition: '==', '!=', '>', '<', '>=', '<=', 'in', or 'not in'.
            * filter[2]: rhs value.

        If `filters` is a list of filters, they are applied sequentially.

        Args:
            filters: A filter or list of filters.
            keep_row: Whether to keep or remove the filtered out rows. If True, the values of the filtered rows are
                set to nan.

        Examples:
            To remove rows, where the value of column 'x' is less than 10,

            >>> panel.filter(('x', '>=', 10))

            This is equivalent to

            >>> panel.data = panel.data[panel.data['x'] >= 10]
        """

        if isinstance(filters[0], str):
            filters = [filters]

        for filter in filters:
            self._filter_data(filter, keep_row)

    def get_row_count(self, ascending=True):
        """Get row count per id.

        The row count is a sequential number starting from 0 for each id. If `ascending` is True (False), its value is
        0 on the first (last) date. It can be used, e.g., to remove first (last) `n` rows of :attr:`data`.

        Args:
            ascending: If True, the row count increases with date.

        Returns:
            Ndarray of row counts.
        """

        if ascending:
            if (self._row_count is None) or (self._row_count.shape[0] != self.data.shape[0]):
                self._row_count = self.get_id_group().cumcount().to_numpy().astype('int32')
            return self._row_count
        else:
            if (self._row_count1 is None) or (self._row_count1.shape[0] != self.data.shape[0]):
                self._row_count1 = self.get_id_group().cumcount(ascending=False).to_numpy().astype('int32')
            return self._row_count1

    def remove_rows(self, data, n=1):
        """Remove rows.

        Remove (set to nan) the first `n` periods of `data` per id. If `n` < 0, the last `-n` periods are
        removed. Data frequency is accounted for: e.g., if ``freq`` = MONTHLY and ``base_freq`` = ANNUAL, `n` = 1
        removes the first 12 rows per id.

        Args:
            data: DataFrame, Series, or ndarray. It should have the same length and order as the ``data`` attribute.
            n: Number of rows (in base frequency) to remove (set to nan).

        Returns:
             The `data` with removed rows set to nan.
        """

        if is_int(data):
            data = data.astype(config.float_type)

        m = int(self.freq / self.base_freq)
        if n >= 0:
            data[self.get_row_count(True) < n * m] = np.nan
        else:
            data[self.get_row_count(False) < -n * m] = np.nan

        return data

    def shift(self, data=None, n=1):
        """Shift data.

        Shift data within each id accounting for data frequency. The shift period is determined by :attr:`freq` and
        :attr:`base_freq`. E.g., if ``freq = MONTHLY`` and ``base_freq = ANNUAL``, `n` = 1 means 1-year shift, and
        `data` is shifted by 12 (12 months). That is, if ``base_freq = ANNUAL (QUARTERLY)``, ``shift(data, n)`` will
        always shift `data` by `n` years (quarters) regardless of the data frequency.

        Args:
            data: DataFrame, Series, ndarray, or (list of) columns. It should have the same length and order as the
                ``data`` attribute. If None, `data` = ``data``.
            n: Shift size in the base frequency.

        Returns:
            Series or DataFrame of shifted data.
        """

        # value = self._to_value(data)
        m = int(self.freq / self.base_freq)

        if data is None:
            return self.get_id_group().shift(m * n)
        elif isinstance(data, list):
            return self.get_id_group()[data].shift(m * n)
        elif isinstance(data, str):
            data = self[data]

        # return self.apply_to_ids(data, shift, None, m * n)
        res = self.remove_rows(np.roll(data, m * n, axis=0), n)  # This returns ndarray.
        return self._to_frame(res, data)

    def diff(self, data, n=1):
        """Get the difference of data.

        Calculate difference within each id accounting for data frequency. The period between two data points is
        determined by :attr:`freq` and :attr:`base_freq`. See :meth:`shift` for details.

        Args:
            data: DataFrame, Series, ndarray, or (list of) columns. It should have the same length and order as the
                ``data`` attribute. If None, `data` = ``data``.
            n: Shift size in the base frequency.

        Returns:
            Series or DataFrame of differenced data.
        """

        if isinstance(data, (str, list)):
            data = self[data]
        return data - self.shift(data, n)

    def pct_change(self, data, n=1, allow_negative_denom=False):
        """Get the percentage change of data.

        Calculate percentage change within each id accounting for data frequency. The period between two data points is
        determined by :attr:`freq` and :attr:`base_freq`. See :meth:`shift` for details.

        Args:
            data: DataFrame, Series, ndarray, or (list of) columns. It should have the same length and order as the
                ``data`` attribute. If None, `data` = ``data``.
            n: Shift size in the base frequency.
            allow_negative_denom: If False, set the output to nan when the denominator is not positive.

        Returns:
            Series or DataFrame of percentage changes.
        """

        # value = self._to_value(data)
        if isinstance(data, (str, list)):
            data = self[data]
        denom = self.shift(data, n)

        if not allow_negative_denom:
            denom[denom <= ZERO] = np.nan
        else:
            denom[denom == 0] = np.nan

        return data / denom - 1

    def cumret(self, ret, period=1, lag=0):
        """Cumulative returns.

        Compute cumulative returns between t-`period` and t-`lag`.
        If `ret` is monthly returns, 12-month momentum can be obtained by setting `period` = 12 and `lag` = 1.
        A negative `period` will generate future returns: e.g., `period` = -1 and `lag` = 0 for one-period ahead
        return; `period` = -3 and `lag` = -1 for two-period ahead
        return starting from t+1.

        Args:
            ret: Series of returns or a return column name. If `ret` is a Series, it should have the same length and
                order as the ``data`` attribute.
            period: Target horizon (in base frequency). (+) for past returns and (-) for future returns.
            lag: Period (in base frequency) to calculate returns from.

        Returns:
            Series of cumulative returns.
        """

        if isinstance(ret, str):
            ret = self.data[ret]

        if period > 0:
            if period == 1 and lag == 0:
                return ret.copy()
            else:
                return np.exp(self.rolling(np.log(ret + 1), period - lag, 'sum', lag=lag)) - 1
        else:
            if period == -1 and lag == 0:
                return self.shift(ret, -1)
            else:
                return np.exp(self.rolling(np.log(ret + 1), -(period - lag), 'sum', lag=period)) - 1

    def futret(self, ret, period=1):
        """Get future returns.

        Args:
            ret: Series of returns or a return column name. If `ret` is a Series, it should have the same length and
                order as the ``data`` attribute.
            period: Target horizon (in base frequency).

        Returns:
            Series of future returns.
        """

        if isinstance(ret, str):
            ret = self.data[ret]

        return self.cumret(ret, -period)

    def rolling(self, data, n, function, min_n=None, lag=0):
        """Apply a function to a rolling window.

        For each id, apply `function` to rolling windows of size `n`. The rolling window is determined by :attr:`freq`
        and :attr:`base_freq`. For instance, if ``freq = MONTHLY`` and ``base_freq = ANNUAL``, `n` = 3 means three-year
        rolling window and the first window consists of the 1st, 13th, and 25th rows.

        Args:
            data: DataFrame, Series, ndarray, or (list of) columns. It should have the same length and order as the
                ``data`` attribute. If None, `data` = ``data``.
            function: Function to apply: 'sum', 'mean', 'std', or 'var'.
            n: Window size in the base frequency.
            min_n: Minimum number of observations in a window. If observations < `min_n`, result is nan. Default to `n`.
            lag: Lag size in the base frequency. The `data` is shifted by `lag` before rolling.

        Returns:
            Series or DataFrame. Rolling calculation result.

        Note:
            For other user-defined functions, use :meth:`apply_to_ids`.

        Examples:
            Suppose ``funda`` is a ``Panel`` object with ``freq = MONTHLY`` and ``base_freq = ANNUAL`` (annual data
            populated monthly), and ``funda.data`` has a return column, 'ret'. The past three-year average return
            starting from one year ago can be calculated (with a condition of at least two non-missing values within
            the sample window) and saved as 'avg_ret3y' as follows:

            >>> funda['avg_ret3y'] = funda.rolling('ret', 'mean', 3, 2, 1)
        """

        min_n = min_n or n

        if lag:
            data = self.shift(data, lag)

        if function == 'sum':
            function = roll_sum
        elif function == 'mean':
            function = roll_mean
        elif function == 'var':
            function = roll_var
        elif function == 'std':
            function = roll_std
        else:
            raise ValueError(f"Unsupported function: {function}. Only 'sum', 'mean', 'var', and 'std' are currently "
                             f"supported.")

        res = self.apply_to_ids(data, function, None, n, min_n)
        return self._to_frame(res, data)

    def rolling_regression(self, data, n, min_n=None, add_const=True):
        """Conduct rolling regression.

        Run rolling OLS for each id accounting for the data frequency.

        Args:
            data: DataFrame, ndarray, or (list of) columns. It should have the same length and order as the
                ``data`` attribute. The first column is used as the dependent variable and the rest as the independent
                variables.
            n: Window size in the base frequency.
            min_n: Minimum number of observations in a window. If observations < `min_n`, result is nan. Default to `n`.
            add_const: If True, add a constant to the independent variables.

        Returns:
            Nx(K+2) ndarray, where N is the length of `data` and K is the number of independent variables.

                * First K columns: Coefficients.
                * K+1-th column: R-squared.
                * K+2-th column: Idiosyncratic volatility (standard deviation of residuals).
        """

        min_n = min_n or n
        n_ret = data.shape[1] + 2  # beta, r2, ivol

        return self.apply_to_ids(data, rolling_regression, n_ret, n, min_n, add_const)

    def copy(self, columns=None, deep=False):
        """Make a copy of this object and return it.

        Args:
            columns: Columns to copy. If None, copy all columns.
            deep: If True, deep copy.

        Returns:
            Copy of this object.
        """

        if not columns:
            return copy.deepcopy(self) if deep else copy.copy(self)
        else:
            new_panel = copy.copy(self)
            new_panel.data = self.data.loc[:, columns].copy(deep)
            return new_panel

    def copy_from(self, panel, columns=None, deep=False):
        """Copy from another Panel object.

        Args:
            panel: A ``Panel`` object to be copied.
            columns: Columns of `panel` to copy. If None, copy all columns.
            deep: If True, deep copy.
        """

        # Copy data.
        if columns:
            self.data = panel.data.loc[:, columns].copy(deep)
        else:
            self.data = panel.data.copy(deep)

        # Copy the other attributes.
        for k, v in panel.__dict__.items():
            if k == 'data':
                continue
            self.__dict__[k] = v

    def save(self, fname=None, fdir=None, columns=None):
        """Save this object to a file.

        The ``data`` attribute is saved to a ``config.file_format`` file and the other attributes are saved to a json
        file.

        Args:
            fname: File name without extension. If None, `fname` = lower-case class name.
            fdir: Directory to save the file. If None, `fdir` = ``config.output_dir``.
            columns: List of columns to save. If None, the entire dataset is saved.
        """

        fname = fname or type(self).__name__.lower()  # lower-case class name
        fdir = fdir or config.output_dir
        fpath = fdir + ('' if fdir[-1] == '/' else '/') + fname + '.json'

        # Save data.
        data = self.data.loc[:, columns] if columns else self.data
        write_to_file(data, fname, fdir)

        # Save the other attributes.
        params = {}
        for k, v in self.__dict__.items():
            try:
                json.dumps(v)
                params[k] = v  # Save only JSON serializable attributes.
            except:
                pass

        with open(fpath, 'w') as fh:
            json.dump(params, fh, indent=4)

    def load(self, fname=None, fdir=None):
        """Load a Panel object from a file.

        Args:
            fname: File name without extension. If None, `fname` = lower-case class name.
            fdir: Directory to load the file from. If None, `fdir` = ``config.output_dir``.
        """

        fname = fname or type(self).__name__.lower()  # lower-case class name
        fdir = fdir or config.output_dir
        fpath = fdir + ('' if fdir[-1] == '/' else '/') + fname + '.json'

        # Load data.
        self.data = read_from_file(fname, fdir)

        # Load the other attributes.
        with open(fpath, 'r') as fh:
            params = json.load(fh)

        for k, v in params.items():
            self.__dict__[k] = v

        return self

    def clean_memory(self):
        """Clean memory.

        Deleting rows/columns of the ``data`` attribute may not release memory immediately. If a ``Panel`` object
        consumes unusually large memory, call this function to release memory.
        """

        for col in self.data:
            self.data[col] = self.data[col]


    ####################################################################################################################
    #
    # DEPRECATED METHODS
    #
    ####################################################################################################################
    def _shift_deprecated(self, data=None, n=1):
        """Shift data.

        This is similar to ``data.shift()`` but the actual shift period is determined by `freq` and `base_freq`.
        E.g., if `freq` = MONTHLY, `base_freq` = ANNUAL, `n` = 1 means 1-year shift, and `data`
        is shifted by 12 (12 months). That is, if `base_freq` = ANNUAL (QUARTERLY), ``shift(data, n)`` will always shift
        `data` by `n` years (quarters) regardless of the data frequency, `freq`.

        Args:
            data: DataFrame, Series, or ndarray. It should have the same length and order as the `data` attribute.
                If `data` is None, ``shift()`` is applied to the `data` attribute in-place.
            n: Shift size in the base frequency.

        Returns:
            Shifted data.
        """

        if data is None:
            self.data = self.shift(self.data, n)
            return

        m = int(self.freq / self.base_freq)
        if isinstance(data, np.ndarray):
            return self.remove_rows(np.roll(data, n * m, axis=0), n)
        else:
            return self.remove_rows(data.shift(n * m), n)
            # return data.groupby(level=-1).shift(n * m)  # Shift data forward.

    def __rolling_deprecated(self, data, n, function, min_n=None, lag=0):
        if lag:
            roll = data.shift(lag).rolling(n, min_periods=min_n)
        else:
            roll = data.rolling(n, min_periods=min_n)

        if function == 'mean':
            return roll.mean().astype(config.float_type)
        elif function == 'std':
            return roll.nanstd().astype(config.float_type)
        elif function == 'var':
            return roll.var().astype(config.float_type)
        elif function == 'sum':
            return roll.sum().astype(config.float_type)
        elif function == 'min':
            return roll.min().astype(config.float_type)
        elif function == 'max':
            return roll.max().astype(config.float_type)

    def _rolling_deprecated(self, data, n, function, min_n=None, lag=0):
        """Apply a function to a rolling window of data.

        This is similar to ``data.rolling()`` but the actual rolling window is determined by `freq` and `base_freq`.
        See :py:meth:`shift` for details.

        Args:
            data: Series or DataFrame. It should have the same index as the `data` attribute.
                If `data` is None, ``rolling()`` is applied to the `data` attribute in-place.
            n: Window size in the base frequency.
            function: Function to apply: 'mean', 'std', 'var', 'sum', 'min', or 'max'.
            min_n: Minimum number of observations in the window. If observations < `min_n`, result is nan.
                If `min_n` = None, `min_n` = `n`.
            lag: Lag size before start rolling.

        Returns:
            Series or DataFrame. Rolling calculation result.

        Examples:
            Suppose `funda` is a ``Panel`` object with `freq` = MONTHLY and `base_freq` = ANNUAL (annual data populated
            monthly), and `funda.data` has a return column, 'ret'. The past three-year average return starting
            from one year ago can be calculated (with a condition of at least two non-missing values within the sample
            window) and saved as 'avg_ret3y' as follows:

            >>> funda.data['avg_ret3y'] = funda.rolling(funda.data['ret'], 3, 'mean', 2, 1)
        """

        if data is None:
            self.data = self.rolling(self.data, n, function, min_n, lag)
            return

        m = int(self.freq / self.base_freq)

        if m == 1:
            return self.remove_rows(self.__rolling_deprecated(data, n, function, min_n, lag), n - 1 + lag)
        else:
            e = len(data)
            res = []
            for i in range(m):
                x = data.iloc[np.arange(i, e, m, dtype=int)]
                res.append(self.__rolling_deprecated(x, n, function, min_n, lag))

            return self.remove_rows(pd.concat(res).sort_index(level=[1, 0]), n - 1 + lag)

    def _rolling_regression_(self, y, X, n, min_n=None, add_constant=True, output=None):
        """Rolling OLS (versitile version).

        Run rolling OLS for each ID accounting for the data frequency.
        This function internally uses ``statsmodels.regression.rolling.RollingOLS()``, which is slow when `output` is
        not None. Use :py:meth:`Panel.rolling_beta` if possible.

        Args:
            y: Series of y. It should have the same index as the `data` attribute.
            X: Series or DataFrame of X. It should have the same index as the `data` attribute.
            n: Window size in the base frequency.
            min_n: Minimum number of observations. See the reference for more details.
            add_constant: If True, add constant to `X`.
            output: List of outputs supported by ``statsmodels.regression.rolling.RollingOLS()``, e.g., 'aic', 'bic',
                and 'fvalue'. See the reference for possible outputs. If None, estimate coefficients only.

        Returns:
            DataFrame of the coefficients if `output` = None, otherwise, DataFrames of the outputs defined by `output`.

        Note:
            ``statsmodels.regression.rolling.RollingOLS()`` is applied to the entire `y` and `X`, i.e., to all IDs,
            and then the first `n-1` rows of each ID's results are set to nan as those results are from the mixed sample
            of the previous ID and the current ID. We adopt this approach instead of applying ``RollingOLS()`` to each
            ID as the latter is significantly slower. One caveat of our approach is that some results can be lost
            when `min_n` < `n`. For instance, if `n` = 5 and `min_n` = 3, the first four results of each ID will be
            removed even though the third and fourth results could be valid.

        References:
            statsmodels.regression.rolling.RollingOLS:
            https://www.statsmodels.org/dev/generated/statsmodels.regression.rolling.RollingOLS.html

            statsmodels.regression.rolling.RollingRegressionResults:
            https://www.statsmodels.org/dev/generated/statsmodels.regression.rolling.RollingRegressionResults.html
        """

        m = int(self.freq / self.base_freq)

        params_only = output is None

        if add_constant:
            X = sm.add_constant(X)

        if m == 1:
            rslt = RollingOLS(y, X, window=n, min_nobs=min_n).fit(params_only=params_only)
            if output is None:
                return rslt.params
            else:
                return tuple([rslt.__getattribute__(arg) for arg in output])
        else:
            e = len(y)
            results = []
            params = []
            for i in range(m):
                y_ = y.iloc[np.arange(i, e, m, dtype=int)]
                X_ = X.iloc[np.arange(i, e, m, dtype=int)]
                rslt = RollingOLS(y_, X_, window=n, min_nobs=min_n).fit(params_only=params_only)
                results.append(rslt)
                params.append(rslt.params)

            if output is None:
                return self.remove_rows(pd.concat(params).sort_index(level=[1, 0]), n - 1)
                # return pd.concat(params).reindex(y.index)
            else:
                retvals = []
                for arg in output:
                    retval = [model.__getattribute__(arg) for model in results]
                    retvals.append(self.remove_rows(pd.concat(retval).sort_index(level=[1, 0]), n - 1))
                    # retvals.append(pd.concat(retval).reindex(y.index))

                return tuple(retvals)


class FCPanel(Panel):
    """Base class for firm characteristic generation.

    For each firm characteristic, there should be one method to generate it and the method name should start with 'c\\_'.
    Generated firm characteristics are added to the ``data`` attribute with column names equal to their method names
    (without 'c\\_').

    Args:
        alias (str, list, or dict): Firm characteristics to generate and their aliases. If None, all available firm
            characteristics are generated.

            * str: A column name in the mapping file (``config.mapping_file_path``). The firm characteristics defined
              in this class and in the `alias` column of the mapping file are generated. See :ref:`sec-mapping file`.
            * list: List of firm characteristics (method names).
            * dict: Dict of firm characteristics and their aliases of the form {method name: alias}.

            If aliases are not specified, method names are used as aliases.
        data: Raw data DataFrame with index = date/id, sorted on id/date.
        freq: Frequency of `data`. ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        base_freq: Frequency of `data` values. ANNUAL, QUARTERLY, MONTHLY, or DAILY.
            For example, if funda is populated monthly, `freq` = MONTHLY and `base_freq` = ANNUAL.
            If None, `base_freq` = `freq`.

    Note:
        Firm characteristics are stored in the ``data`` attribute with column names equal to their method
        names (without 'c\\_'). When a ``FCPanel`` is saved to a file using :meth:`save`, the firm characteristic
        columns are renamed by the aliases , and when a saved ``FCPanel`` is loaded using :meth:`load`,
        the firm characteristic columns are renamed by the method names.

    **Attributes**

    Attributes:
        data: DataFrame with index = date/id, sorted on id/date that stores a panel data.
        freq: Frequency of ``data``. ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        base_freq: Frequency of ``data`` values. ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        char_map: Dictionary to map firm characteristic generation methods with aliases: ``char_map['method']`` returns
            the alias.

    **Methods**

    .. autosummary::
        :nosignatures:

        show_available_chars
        get_available_chars
        get_char_list
        create_chars
        created
        prepare
        rename_chars
        remove_rawdata
        save
        load
    """

    def __init__(self, alias=None, data=None, freq=MONTHLY, base_freq=None):
        super().__init__(data, freq, base_freq)

        if isinstance(alias, str):
            mapping = pd.read_excel(config.mapping_file_path)
            mapping.dropna(subset=[alias], inplace=True)
            self.char_map = mapping.set_index('function', drop=False)[alias].to_dict()
        elif isinstance(alias, dict):
            self.char_map = alias
        elif isinstance(alias, list):
            self.char_map = {char: char for char in alias}
        else:
            self.char_map = {fcn[2:]: fcn[2:] for fcn in dir(self) if callable(getattr(self, fcn)) and fcn[:2] == 'c_'}

    def show_available_chars(self, all=False):
        """Display firm characteristics available in this class.

        Args:
            all: If True, display all firm characteristics available in this class, otherwise, display only the firm
                characteristics to generate.
        """

        fcns = {fcn[2:]: getattr(self, fcn).__doc__ for fcn in dir(self)
                if callable(getattr(self, fcn)) and fcn[:2] == 'c_'}

        if not all:
            chars = self.get_char_list()
            fcns = {k: v for k, v in fcns.items() if k in chars}

        if not fcns:
            print('There are no characteristic generation functions in this class.')
            return

        s = pd.Series(fcns)
        max_rows = pd.get_option('display.max_rows')  # get current setting
        pd.set_option('display.max_rows', len(s))  # set to len(s) so that all functions are printed.
        print(s.str.ljust(int(s.str.len().max())))
        pd.set_option('display.max_rows', max_rows)  # back to previous setting

    def get_available_chars(self):
        """Get all firm characteristics available in this class.

        Returns:
            List of characteristics (method names)
        """

        return [fcn[2:] for fcn in dir(self) if callable(getattr(self, fcn)) and fcn[:2] == 'c_']

    def get_char_list(self):
        """Get the firm characteristics to generate.

        Among all the firm characteristics to generate, the firm characteristics defined in this class are returned.

        Returns:
            A dictionary of firm characteristics with keys equal to method names and values equal to aliases.
        """

        fcn_list = self.get_available_chars()
        char_list = {k: v for k, v in self.char_map.items() if k in fcn_list}
        return char_list

    def create_chars(self, char_list=None):
        """Generate firm characteristics.

        Generated firm characteristics are added to :attr:`data` using their method names as the column names.

        Args:
            char_list: List or dict of firm characteristics (method names) to generate. If dict, keys should be method
             names and values aliases. If None, all firm characteristics available in this class and specified by the
             `alias` argument are generated.
        """

        # Add firm characteristic columns to data.
        if not char_list:
            char_list = self.get_char_list()
        elif isinstance(char_list, str):
            if char_list in self.get_available_chars():
                char_list = {char_list: char_list}
        elif isinstance(char_list, list):
            char_list = {v: v for v in char_list if v in self.get_available_chars()}
        elif isinstance(char_list, dict):
            char_list = {k: v for k, v in char_list.items() if k in self.get_available_chars()}

        columns = [char for char in char_list if char not in self.data.columns]
        if columns:
            self.data[columns] = np.nan
            if config.float_type == 'float32':
                for col in columns:
                    self.data[col] = self.data[col].astype(config.float_type)

        # Create firm characteristics.
        for fcn in char_list:
            try:
                if not self.created(fcn):
                    # retval = eval(f'self.c_{fcn}()')
                    # self.data.loc[:, fcn] = retval.to_numpy() if isinstance(retval, pd.Series) else retval
                    self.data[fcn] = eval(f'self.c_{fcn}()').astype(config.float_type)
                elapsed_time(f"[{fcn} ({char_list[fcn]})] created.")
            except Exception as e:
                log(f'Error occurred while creating {fcn}: {e}')
                raise e

    def created(self, char):
        """Check if a firm characteristic has already been created.

        Args:
            char: Firm characteristic (the method name without 'c\\_') to check.

        Returns:
            True if the firm characteristic has been created.
        """

        return (char in self.data) and (not self.data[char].isna().all())

    def prepare(self, char_list):
        """Prepare characteristics that are required to generate a characteristic.

        If a characteristic is defined as a function of other characteristics, this function will check whether they
        already exist and generate them if they don't.

        Args:
            char_list: list of firm characteristics (method names without 'c\\_') to prepare.
        """

        for fcn in char_list:
            if not self.created(fcn):
                self.create_chars([fcn])

    def rename_chars(self, to_alias=True):
        """Rename the firm characteristic columns.

        Args:
            to_alias: If True, rename firm characteristic columns from method names to aliases, and vice versa.
        """

        if to_alias:  # method -> alias
            map = {k: v for k, v in self.char_map.items() if k in self.data}
        else:  # alias -> method
            map = {v: k for k, v in self.char_map.items() if v in self.data}

        self.data.rename(columns=map, inplace=True)

    def remove_rawdata(self, excl_columns=None):
        """Remove raw data.

        Raw data columns of :attr:`data` except `excl_columns` are deleted. If raw data are not needed after generating
        firm characteristics, calling this method can reduce memory and hard disc usage.

        Args:
            excl_columns: List of columns to exclude.
        """

        columns = unique_list(list(self.char_map) + list(self.char_map.values()) + (excl_columns or []))
        keep_columns(self.data, columns)
        self.clean_memory()

    def save(self, fname=None, fdir=None, other_columns='all'):
        """Save this object to a file.

        The ``data`` attribute is saved to a ``config.file_format`` file and the other attributes are saved to a json
        file. Firm characteristic columns are renamed by aliases before saving.

        Args:
            fname: File name without extension. If None, `fname` = lower-case class name.
            fdir: Directory to save the file. If None, `fdir` = ``config.output_dir``.
            other_columns: List of columns other than firm characteristic columns to save. If None, only firm
                characteristics are saved; if 'all', all columns are saved; if list, firm characteristic columns plus
                `other_columns` are saved.
        """

        # Choose columns to save.
        char_columns = [v for k, v in self.char_map.items() if k in self.data]
        if not other_columns:
            columns = char_columns
        elif other_columns == 'all':
            columns = None
        else:
            columns = other_columns + char_columns

        self.rename_chars(True)
        super().save(fname, fdir, columns)
        self.rename_chars(False)

    def load(self, fname=None, fdir=None):
        """Load a FCPanel object from a file.

        Firm characteristic columns are renamed by method names after loading.

        Args:
            fname: File name without extension. If None, `fname` = lower-case class name .
            fdir: Directory to read the file from. If None, `fdir` = ``config.output_dir``.
        """

        super().load(fname, fdir)
        self.rename_chars(False)
        return self


if __name__ == '__main__':
    os.chdir('../')

    df = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    print(df)
    panel = Panel(df)
    print(panel['a'])
    print(panel[:, 'a'])
    print(panel[['a', 'b']])
    print(panel[1, 'b'])
    panel[1, 'b'] = 100
    panel['c'] = [200, 200]
