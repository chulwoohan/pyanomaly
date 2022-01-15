"""This module defines `Panel` class, which serves as the base class for panel data analysis."""

import numpy as np
import pandas as pd
import json
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from pyanomaly.globals import *
from pyanomaly.datatools import *
from pyanomaly.analytics import rolling_beta
from pyanomaly.fileio import write_to_file, read_from_file


class Panel:
    """Base class for panel data.

    Data is stored in the attribute `data`. `data` should have index = date/id, i.e., the first index should be
    time-series identifier and the second index should be cross-section identifier, and always be sorted on id/date.
    The date index must be of datetime type.

    Args:
        alias: Characteristic column name in ``mapping.xlsx``. If None, function names (without 'c\_') are used as the
            characteristic names.
        data: DataFrame with index = date/id, sorted on id/date.
        freq: Frequency of `data`. ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        base_freq: Frequency of `data` values. funda: ANNUAL, fundq: QUARTERLY, crspm: MONTHLY, crspd: DAILY.
            For example, if funda is populated monthly, `freq` = MONTHLY and `base_freq` = ANNUAL.
            If None, `base_freq` = `freq`.

    Attributes:
        data: DataFrame with index = date/id, sorted on id/date that stores a panel data.
        freq: Frequency of `data`. ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        base_freq: Frequency of `data` values. ANNUAL, QUARTERLY, MONTHLY, or DAILY.
        char_map: Dictionary to map functions with aliases. `char_map['alias']` returns its corresponding function name.
        reverse_map: Function to alias mapping. `reverse_map['function']` returns its corresponding alias.
    """

    def __init__(self, alias=None, data=None, freq=MONTHLY, base_freq=None):
        self.data = data
        self.freq = freq
        self.base_freq = base_freq or freq

        # alias -> function
        alias_ = alias or 'function'
        char_map = pd.read_excel(config.mapping_file_path)
        char_map.dropna(subset=[alias_], inplace=True)
        self.char_map = char_map.set_index(alias_, drop=False)['function'].to_dict()

        if alias is None:  # Add new characteristic functions not in the mapping file yet.
            char_fncs = [fcn[2:] for fcn in dir(self) if callable(getattr(self, fcn)) and fcn[:2] == 'c_']
            for fcn in char_fncs:
                self.char_map[fcn] = fcn

        # function -> alias
        self.reverse_map = {v: k for k, v in self.char_map.items()}

    def rename(self, to_alias=True):
        """Rename `data` columns.

        Args:
            to_alias: If True, rename firm characteristic columns from function names to aliases, and vice-versa.
        """

        if to_alias:  # function -> alias
            map = {col: self.reverse_map[col] for col in self.data if col in self.reverse_map}
        else:  # alias -> function
            map = {col: self.char_map[col] for col in self.data if col in self.char_map}

        self.data.rename(columns=map, inplace=True)

    def date_col(self):
        """Get date index name."""

        return self.data.index.names[0]

    def id_col(self):
        """Get ID index name."""

        return self.data.index.names[-1]

    def date_values(self):
        """Get date index values."""

        return self.data.index.get_level_values(0)

    def id_values(self):
        """Get ID index values."""

        return self.data.index.get_level_values(-1)

    def inspect_data(self, option=['summary']):
        """Inspect data.

        See ``datatools.inspect_data()`` for details.
        """

        inspect_data(self.data, option)

    def save(self, fname=None, fdir=None, other_columns=None):
        """Save this object.

        Parameters are saved to a json file and `data` is saved to a pickle file.

        Args:
            fname: File name without extension. If None, `fname` = class name.
            fdir: Directory to save the file. If None, `fdir` = `config.output_dir`.
            other_columns: List of columns to save other than firm characteristics. If given, characteristic columns
                plus `other_columns` are saved. If None, all columns are saved. Note that unsaved columns will be deleted
                from `self.data`.
        """

        fname = fname or type(self).__name__.lower()  # lower-case class name
        fdir = fdir or config.output_dir
        fpath = fdir + ('' if fdir[-1] == '/' else '/') + fname + '.json'

        # Save data.
        if other_columns is not None:
            columns = other_columns + list(self.reverse_map.keys())
            keep_columns(self.data, columns)

        self.rename()  # To aliases.
        write_to_file(self.data, fname, fdir)
        self.rename(False)  # Back to function names.

        # Save parameters.
        params = {
            'freq': self.freq,
            'base_freq': self.base_freq,
            'char_map': self.char_map,
            'reverse_map': self.reverse_map,
        }
        with open(fpath, 'w') as fh:
            json.dump(params, fh, indent=4)

    def load(self, fname=None, fdir=None):
        """Load this object from a file.

        Args:
            fname: File name without extension. If None, `fname` = class name.
            fdir: Directory to read the file from. If None, `fdir` = `config.output_dir`.
        """

        fname = fname or type(self).__name__.lower()  # lower-case class name
        fdir = fdir or config.output_dir
        fpath = fdir + ('' if fdir[-1] == '/' else '/') + fname + '.json'

        # Load parameters.
        with open(fpath, 'r') as fh:
            params = json.load(fh)

        self.freq = params['freq']
        self.base_freq = params['base_freq']
        self.char_map = params['char_map']
        self.reverse_map = params['reverse_map']

        # Load data.
        self.data = read_from_file(fname, fdir)
        self.rename(False)

    def _filter_data(self, filter, keep_row=False):
        column = filter[0]
        condition = filter[1]
        value = filter[2]

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
            raise ValueError(f'Unrecognized condition: {condition}')

    def filter(self, filters, keep_row=False):
        """Filter data using the conditions defined by `filters`.

        A filter is a tuple of three elements:

            * filter[0]: column to apply the filter to.
            * filter[1]: filter condition: '==', '!=', '>', '<', '>=', '<=', 'in', or 'not in'.
            * filter[2]: rhs value.

        If `filters` is a list of filters, they are applied sequentially.

        For example, to remove `data['x'] < 10`,

            >>> panel.filter_data(('x', '>=', 10))

        This is equivalent to

            >>> panel.data = panel.data[panel.data['x'] >= 10]

        Args:
            filters: A filter or list of filters.
            keep_row: Whether to keep or remove the filtered out rows. If True, the values of the filtered rows are
                set to None.
        """

        if isinstance(filters[0], str):
            filters = [filters]

        for filter in filters:
            self._filter_data(filter, keep_row)

    def show_available_functions(self):
        """Display all functions for firm characteristics."""

        fcns = {fcn[2:]: getattr(self, fcn).__doc__ for fcn in dir(self) if callable(getattr(self, fcn)) and fcn[:2] == 'c_'}
        s = pd.Series(fcns)
        max_rows = pd.get_option('display.max_rows')  # get current setting
        pd.set_option('display.max_rows', len(s)) # set to len(s) so that all functions are printed.
        print(s.str.ljust(int(s.str.len().max())))
        pd.set_option('display.max_rows', max_rows)  # back to previous setting

    def get_available_chars(self):
        """Get the list of available characteristics.

        This function returns the aliases of the characteristics defined in `alias` column of ``mapping.xlsx``.

        Returns:
            A list of characteristics.
        """

        fcn_list = [fcn[2:] for fcn in dir(self) if callable(getattr(self, fcn)) and fcn[:2] == 'c_']
        char_list = [self.reverse_map[fcn] for fcn in fcn_list if fcn in self.reverse_map]
        return char_list

    def create_chars(self, char_list=None):
        """Generate firm characteristics.

        Created firm characteristics are added to ``data`` using their function names as the column names.

        Args:
            char_list: List of characteristics (aliases) to create. If None, all available characteristics are created.
        """

        char_list = char_list or self.get_available_chars()

        for i, char in enumerate(char_list):
            try:
                fcn = self.char_map[char]
                self.data[fcn] = eval(f'self.c_{fcn}()')
                elapsed_time(f"[{char}] created.")
                if i and (i % 30 == 0):
                    self.data = self.data.copy()
            except Exception as e:
                log(f'Error occured while creating {char}: {e}')
                raise e

    def prepare(self, char_fcns):
        """Prepare "ingredient" characteristics that are required to generate a characteristic.

        If a characteristic is defined as a function of other characteristics, this function will check whether they
        already exist and generate them if they do not exist.

        Args:
            char_fcns: list of function names (without 'c\_') of the ingredient characteristics.
        """

        for fcn in char_fcns:
            if fcn not in self.data:
                self.create_chars([self.reverse_map[fcn]])

    def add_row_count(self):
        """Add row count column ('rcount`) to `data`.

        The rcount has values 0, 1, ... for each ID starting from the first date.
        This can be use to remove the first n rows of each ID.

        NOTE:
            If the rows of `data` is changed, e.g., by filtering, this function should be called again as
            rcount may no longer be continuous.
        """

        self.data['rcount'] = self.data.groupby(self.id_col()).cumcount()

    def remove_rows(self, data, nrow=1):
        """Remove (set to nan) the first `nrow` rows of `data` for each ID.

        Args:
            data: Dataframe, Series, or ndarray. Its length should be the same as the length of `self.data`.
            nrow: Number of rows to remove (set to nan).

        Returns:
             `data` with rows removed.
        """

        m = int(self.freq / self.base_freq)
        data[self.data.rcount.values < nrow * m] = np.nan
        return data

    def populate(self, freq, method='ffill', limit=None):
        """Populate data.

        See ``datatools.populate()`` for details.
        """

        self.data = populate(self.data, freq, method, limit)
        self.freq = freq


    def copy(self, columns=None, deep=False):
        """Make a copy of this object and return it.

        Args:
            columns: Columns of `data` to copy.
            deep: If True, copy `data`, otherwise, only the address of `data` is copied.

        Returns:
            Copy of this object.
        """

        copy = Panel()
        copy.copy_from(self, columns, deep)
        return copy

    def copy_from(self, panel, columns=None, deep=False):
        """Copy `panel` to this object.

        Args:
            panel: `Panel` object to be copied.
            columns: Columns of `panel.data` to copy.
            deep: If True, copy `panel.data`, otherwise, only the address of `panel.data` is copied.
        """

        columns = columns or panel.data.columns
        self.data = panel.data[columns].copy(deep) if deep else panel.data[columns]
        self.freq = panel.freq
        self.base_freq = panel.base_freq
        self.char_map = panel.char_map
        self.reverse_map = panel.reverse_map

    def merge(self, right, on=None, how='inner', drop_duplicates='right'):
        """Merge `data` with `right`.

        Unlike `pd.merge()`, the index of `data` is always retained.

        Args:
            right: Series or Dataframe to merge with
            on: Columns to merge on. If None, merge on index.
            how: Merge method: 'inner', 'outer', 'left', or 'right'.
            drop_duplicates: how to handle duplicate columns. 'left': keep right, 'right': keep left,
                None: keep both. If None, '_x' ('_y') are added to the duplicate column names of left (right).
        """

        data = self.data
        index_names = data.index.names

        if on is None:
            is_index = True
            on = index_names
        elif isinstance(on, str):
            is_index = on in index_names
        else:
            is_index = False
            for i in on:
                is_index |= i in index_names

        if is_index:
            data.reset_index(inplace=True)

        dup_columns = data.columns.intersection(right.columns).difference(on)
        if drop_duplicates == 'right':
            self.data = data.merge(right[right.columns.difference(dup_columns)], on=on, how=how)
        elif drop_duplicates == 'left':
            self.data = data[data.columns.difference(dup_columns)].merge(right, on=on, how=how)
        else:
            self.data = data.merge(right, on=on, how=how)

        if is_index:
            self.data.set_index(index_names, inplace=True)

    def cumret(self, ret, period=1, lag=0):
        """Compute cumulative returns between t-`period` and t-`lag`.

        E.g., 12-month momentum can be obtained by setting `period` = 12 and `lag` = 1.
        A negative `period` will generate future returns: eg, `period` = -1 and `lag` = 0 for one-period ahead
        return; `period` = -3 and `lag` = -1 for two-period ahead
        return starting from t+1.

        NOTE:
            Returns `ret` should be in `base_freq`. If `base_freq` = ANNUAL, the returns in `ret` should be annual
            returns regardless of the value of `freq`.

        Args:
            ret: Series of returns in `base_freq`.
            period: Target horizon. (+) for past returns and (-) for future returns.
            lag: Period to calculate returns from.

        Returns:
            Series of cumulative returns.
        """

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
        """Get `period`-period ahead return.
        """

        return self.cumret(ret, -period)

    def sum(self, *args):
        """Sum of `args`.

        This is similar to ``sum()`` of SAS: nan's of `args` are replaced by 0.

        Args:
            args: List of Series.

        Returns:
            (Series) Sum of `args`
        """

        y = np.zeros_like(args[0])
        isnan = True
        for x in args:
            y += x.fillna(0)
            isnan &= x.isna()
        y[isnan] = np.nan

        return y

    def shift(self, data, n=1):
        """Shift `data` accounting for frequency.

        This is similar to ``data.shift()`` but the actual shift period is determined by `freq` and `base_freq`.
        E.g., if `freq` = MONTHLY, `base_freq` = ANNUAL, `n` = 1 means 1-year shift, and `data`
        is shifted by 12 (12 months). That is, if `base_freq` = ANNUAL (QUARTERLY), ``shift(n)`` will always shift data
        by `n` years (quarters) regardless of the data frequency, `freq`.

        Args:
            data: Series or Dataframe.
            n: Shift size in terms of base frequency.

        Returns:
            Shifted data.
        """

        m = int(self.freq / self.base_freq)
        return self.remove_rows(data.shift(n * m), n)

    def pct_change(self, data, n=1, allow_negative_denom=False):
        """Get the percentage change of `data` accounting for frequency.

        This is similar to ``data.pct_change()`` but the actual period between two data points is determined by `freq` and `base_freq`.
        See ``Panel.shift()`` for details.

        Args:
            data: Series or Dataframe.
            n: Shift size in terms of base frequency.
            allow_negative_denom: If False, set the output to nan when the denominator is not positive.

        Returns:
            Series or Dataframe of percentage changes.
        """

        denom = self.shift(data, n)
        if not allow_negative_denom:
            denom[denom <= ZERO] = np.nan
        else:
            denom[denom == 0] = np.nan

        return data / denom - 1

    def diff(self, data, n=1):
        """Get the difference of the elements of `data` accounting for frequency.

        This is similar to ``data.diff()`` but the actual period between two data points is determined by `freq` and `base_freq`.
        See ``Panel.shift()`` for details.

        Args:
            data: Series or Dataframe.
            n: Shift size in terms of base frequency.

        Returns:
            Differenced data.
        """

        m = int(self.freq / self.base_freq)
        return self.remove_rows(data.diff(n * m), n)

    def _rolling(self, data, n, function, min_n=None, lag=0):
        if lag:
            roll = data.shift(lag).rolling(n, min_periods=min_n)
        else:
            roll = data.rolling(n, min_periods=min_n)

        if function == 'mean':
            return roll.mean()
        elif function == 'std':
            return roll.std()
        elif function == 'var':
            return roll.var()
        elif function == 'sum':
            return roll.sum()
        elif function == 'min':
            return roll.min()
        elif function == 'max':
            return roll.max()

    def rolling(self, data, n, function, min_n=None, lag=0):
        """Apply `function` to  a rolling window of `data`.

        This is similar to ``data.rolling()`` but the actual rolling window is determined by `freq` and `base_freq`.
        See ``Panel.shift()`` for details.

        Args:
            data: Series or Dataframe.
            n: Window size.
            function: Function to apply: 'mean', 'std', 'var', 'sum', 'min', or 'max'.
            min_n: Minimum number of observations in the window. If observations < `min_n`, result is nan.
                If `min_n` = None, `min_n` = `n`.
            lag: Lag size before start rolling.

        Returns:
            (Series or Dataframe) Rolling calculation result.

        Examples:
            Suppose `funda` is a `Panel` object with `freq` = MONTHLY and `base_freq` = ANNUAL (annual data populated
            monthly), and `funda.data` has a return column, 'ret'. The past three-year average return starting
            from one year ago can be calculated (with a condition of at least two non-missing values within the sample
            window) and saved as 'avg_ret3y' as follows:

            >>> funda.data['avg_ret3y'] = funda.rolling(funda.data['ret'], 3, 'mean', 2, 1)
        """

        m = int(self.freq / self.base_freq)
        if m == 1:
            return self.remove_rows(self._rolling(data, n, function, min_n, lag), n-1+lag)
        else:
            e = len(data)
            res = []
            for i in range(m):
                x = data.iloc[np.arange(i, e, m, dtype=int)]
                res.append(self._rolling(x, n, function, min_n, lag))

            return self.remove_rows(pd.concat(res).sort_index(level=[1, 0]), n-1+lag)

    def rolling_regression(self, y, X, n, min_n=None, add_constant=True, output=None):
        """Rolling OLS accounting for frequency.

        This function internally uses ``statsmodels.RollingOLS()``, which is slow when `output` is not None.
        Use ``Panel.rolling_beta()`` if possible.

        NOTE:
            When this function is used, the first `n-1` rows of each ID should be removed. This means that when
            `min_n` < `n`, some results can be lost. E.g., if `n` = 5 and `min_n` = 3, the rows with `data.rcount` < 4
            should be removed, and the results for `data.rcount` = 3, 4 will be lost.

        Args:
            y: Series of y with index = date/id.
            X: Series or Dataframe of X with index = date/id.
            n: Window size in terms of base frequency.
            min_n: Minimum number of observations in the window. If observations < `min_n`, result is nan.
                If `min_n` = None, `min_n` = `n`.
            add_constant: If True, add constant to `X`.
            output: List of outputs supported by ``statsmodels.RollingOLS()``. If None, estimate coefficients only.

        Returns:
            DataFrame of the coefficients if `output` = None, otherwise, DataFrames of the outputs defined by `output`.
        """

        m = int(self.freq / self.base_freq)

        params_only = output is None

        if add_constant:
            X = sm.add_constant(X)

        if m == 1:
            model = RollingOLS(y, X, window=n, min_nobs=min_n).fit(params_only=params_only)
            if output is None:
                return model.params
            else:
                return tuple([model.__getattribute__(arg) for arg in output])
        else:
            e = len(y)
            models = []
            params = []
            for i in range(m):
                y_ = y.iloc[np.arange(i, e, m, dtype=int)]
                X_ = X.iloc[np.arange(i, e, m, dtype=int)]
                model = RollingOLS(y_, X_, window=n, min_nobs=min_n).fit(params_only=params_only)
                models.append(model)
                params.append(model.params)

            if output is None:
                return self.remove_rows(pd.concat(params).sort_index(level=[1, 0]), n-1)
                # return pd.concat(params).reindex(y.index)
            else:
                retvals = []
                for arg in output:
                    retval = [model.__getattribute__(arg) for model in models]
                    retvals.append(self.remove_rows(pd.concat(retval).sort_index(level=[1, 0]), n-1))
                    # retvals.append(pd.concat(retval).reindex(y.index))

                return tuple(retvals)

    def rolling_beta(self, data, n, min_n=None):
        """Rolling OLS accounting for frequency.

        This is similar to ``Panel.rolling_regression()``: this function is faster but the output is limited to
        coefficients, R2, and idiosyncratic volatility.

        Args:
            data: DataFrame with index = date/id, sorted on id/date. The first column must be y and the rest X
                (not including constant).
            n: Window size in terms of base frequency.
            min_n: Minimum number of observations in the window. If observations < `min_n`, result is nan.
                If `min_n` = None, `min_n` = `n`.

        Returns:
            Coefficients, R2, idiosyncratic volatility. These are NxK, Nx1, and Nx1 ndarrays, respectively,
            where N = `len(data)` and K = number of X + 1.
        """

        m = int(self.freq / self.base_freq)

        if m == 1:
            return rolling_beta(data, n, min_n)
        else:
            e = data.shape[0]
            idx = []
            beta = []
            r2 = []
            idio = []
            for i in range(m):
                idx_ = np.arange(i, e, m, dtype=int)
                data_ = data.iloc[idx_]
                beta_, r2_, idio_ = rolling_beta(data_, n, min_n)
                idx.append(idx_)
                beta.append(beta_)
                r2.append(r2_)
                idio.append(idio_)

            idx = np.argsort(np.concatenate(idx))
            beta = np.concatenate(beta)[idx]
            r2 = np.concatenate(r2)[idx]
            idio = np.concatenate(idio)[idx]

            return beta, r2, idio


