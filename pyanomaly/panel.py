import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from pyanomaly.globals import *
from pyanomaly.datatools import *
from pyanomaly.analytics import rolling_beta
from pyanomaly.fileio import write_to_file, read_from_file


class Panel:
    """Base class for panel data"""

    def __init__(self, acronym=None, data=None, freq=MONTHLY, base_freq=None):
        self.data = data  # Dataframe with index=date/id
        self.freq = freq  # data frequency. Eg, if funda is populated monthly, freq=MONTHLY.
        self.base_freq = base_freq or freq  # funda: ANNUAL, fundq: QUARTERLY, crspm: MONTHLY, crspd: DAILY
        self.acronym = acronym or 'function' # variable (characteristic) column name in mapping.xlsx. If None, function
                                             # names are used as the characteristic names.

        char_map = pd.read_excel(config.input_dir + 'mapping.xlsx')
        char_map.dropna(subset=[self.acronym], inplace=True)
        self.char_map = char_map.set_index(self.acronym, drop=False)['function'].to_dict()

    def date_col(self):
        """Date index name"""
        return self.data.index.names[0]

    def id_col(self):
        """ID index name"""
        return self.data.index.names[-1]

    def date_values(self):
        """Date index values"""
        return self.data.index.get_level_values(0)

    def id_values(self):
        """ID index values"""
        return self.data.index.get_level_values(-1)

    def inspect_data(self, data=None, option=['summary']):
        """Inspect data."""
        if data is None:
            data = self.data
        inspect_data(data, option)

    def save_data(self, fname=None, fdir=None, data=None):
        """save data to file."""
        if data is None:
            data = self.data
        fname = fname or type(self).__name__.lower()  # lower-case class name
        write_to_file(data, fname, fdir)

    def load_data(self, fname=None, fdir=None):
        """Load data from file."""
        fname = fname or type(self).__name__.lower()  # lower-case class name
        self.data = read_from_file(fname, fdir)

    def filter_data_(self, filter):
        """Filter data. For example, if filter={'column': 'X', 'condition': '>=', 'value': 10}, data is filter as
        follows:
            data = data[data['X'] >= 10]

        Args:
            filter: dictionary with three keys:
            'column': column to apply the filter
            'condition': filter condition.
            'value': rhs value.
        """
        column = filter['column']
        condition = filter['condition']
        value = filter['value']

        if condition == '==':
            self.data = self.data[self.data[column] == value]
        elif condition == '!=':
            self.data = self.data[self.data[column] != value]
        elif condition == '>':
            self.data = self.data[self.data[column] > value]
        elif condition == '<':
            self.data = self.data[self.data[column] < value]
        elif condition == '>=':
            self.data = self.data[self.data[column] >= value]
        elif condition == '<=':
            self.data = self.data[self.data[column] <= value]
        elif condition == 'in':
            self.data = self.data[self.data[column].isin(value)]
        elif condition == 'not in':
            self.data = self.data[~self.data[column].isin(value)]
        else:
            raise ValueError(f'Unrecognized condition: {condition}')

    def filter_data(self, filters):
        """Filter data using a list of filters.

        Args:
            filter: filter or list of filters.
        """
        if isinstance(filters, dict):
            filters = [filters]

        for filter in filters:
            self.filter_data_(filter)

    def inverse_map(self, fcns):
        """Get characteristic names mapped to functions.

        Args:
            fcns: (A list of) function name(s).

        Returns:
            (A list of) characteristic name(s).
        """
        if isinstance(fcns, str):
            return list(self.char_map)[list(self.char_map.values()).index(fcns)]
        else:
            return [list(self.char_map)[list(self.char_map.values()).index(fcn)] for fcn in fcns if fcn in list(self.char_map.values())]

    def show_available_functions(self):
        """Show all functions for characteristics."""
        fcns = {fcn[2:]: getattr(self, fcn).__doc__ for fcn in dir(self) if callable(getattr(self, fcn)) and fcn[:2] == 'c_'}
        s = pd.Series(fcns)
        max_rows = pd.get_option('display.max_rows')  # get current setting
        pd.set_option('display.max_rows', len(s)) # set to len(s) so that all functions are printed.
        print(s.str.ljust(int(s.str.len().max())))
        pd.set_option('display.max_rows', max_rows)  # back to previous setting

    def get_available_chars(self):
        """Return the list of available characteristics, ie, those whose functions are defined."""
        fcn_list = [fcn[2:] for fcn in dir(self) if callable(getattr(self, fcn)) and fcn[:2] == 'c_']
        char_list = self.inverse_map(fcn_list)
        return char_list

    def create_chars(self, char_list=None):
        """Create characteristics. Characteristics are added to self.data.

        Args:
            char_list: list of characteristics to create. If None, all available characteristics are created.
        """
        char_list = char_list or self.get_available_chars()

        for char in char_list:
            try:
                fcn = self.char_map[char]
                self.data[char] = eval(f'self.c_{fcn}()')
                elapsed_time(f"[{char}] created.")
            except Exception as e:
                log(f'Error occured while creating {char}: {e}')

    def _prepare(self, char_fcns):
        """Prepare "ingredient" characteristics that are required to calculate a characteristic. If the ingredients
        don't exist, they will be created.

        Args:
            char_fcns: list of function names (without c_) for the ingredient characteristics.
        """
        char_list = self.inverse_map(char_fcns)
        for col in char_list:
            if col not in self.data:
                self.create_chars([col])

    def add_row_count(self):
        """Assign row count for each ID."""
        self.data['rcount'] = self.data.groupby(self.id_col()).cumcount()

    def remove_rows(self, data, nrow=1):
        """Remove (set to nan) the first 'nrow' rows of data per ID.

        Args:
            data: Dataframe, Series, or ndarray.
            nrow: number of rows to delete (set to nan).

        Returns:
             data with rows removed.
        """

        m = int(self.freq / self.base_freq)
        data[self.data.rcount.values < nrow * m] = np.nan
        return data

    def populate(self, freq, method='ffill', limit=None):
        """Populate data. See datatools.populate for details."""

        self.data = populate(self.data, freq, method, limit)
        self.freq = freq


    def copy(self, panel, columns=None, copy_data=False):
        """Copy data from panel.

        Args:
            panel: Panel class to copy from.
            columns: columns to copy.
            copy_data: copy data if True.
        """
        columns = columns or panel.data.columns
        self.data = panel.data[columns].copy() if copy_data else panel.data[columns]
        self.freq = panel.freq
        self.base_freq = panel.base_freq
        self.acronym = panel.acronym
        self.char_map = panel.char_map

    def merge(self, right, on=None, how='inner', drop_duplicates='right'):
        """Merge self.data with right and set the result to self.data. Unlike pd.merge, the index of self.data is
        always retained.

        Args:
            right: Series or Dataframe to merge with
            on: columns to merge on. If None, merge on index.
            how: merge method: 'inner', 'outer', 'left', 'right'.
            drop_duplicates: how to handle duplicate columns. 'left': keep right, 'right': keep left, None: keep both (
            a duplicate column col becomes col_x (left) and col_y (right)).
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

    def sum(self, *args):
        """Sum of a and b. If either a or b is None, set it to 0. Equivalent to sum(a, b) of SAS."""
        y = np.zeros_like(args[0])
        isnan = True
        for x in args:
            y += x.fillna(0)
            isnan &= x.isna()
        y[isnan] = np.nan
        return y

    def shift(self, data, n=1):
        """Shift data accounting for frequency. Eg, if freq=MONTHLY, base_freq=ANNUAL, n=1 means 1-year shift, and data
        is shifted by 12 (12 months).

        Args:
            data: Series or Dataframe to shift.
            n: shift size in terms of base frequency.

        Returns:
            Shifted data.
        """
        m = int(self.freq / self.base_freq)
        return data.shift(n * m)

    def pct_change(self, data, n=1, allow_negative_denom=False):
        """Percentage change of data accounting for frequency.

        Args:
            data: Series or Dataframe to shift.
            n: shift size in terms of base frequency.
            allow_negative_denom: If False, set the output to nan when the denominator is not positive.

        Returns:
            Series or Dataframe of percentage changes.
        """
        denom = self.shift(data, n)
        if not allow_negative_denom:
            denom[denom < 1e-5] = np.nan
        return data / denom - 1

    def diff(self, data, n=1):
        """The same as Dataframe.diff but accounting for frequency."""
        m = int(self.freq / self.base_freq)
        return data.diff(n * m)

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
        """Similar to Dataframe.rolling but accounting for frequency.
        
        Args:
            data: Series or Dataframe to apply rolling.
            n: window size.
            function: name of function to apply, eg, 'mean', 'std', 'sum'. See _rolling for the complete list.
            min_n: minimum number of observations in the window. If observations < min_n, result is nan.
                   If min_n = None, min_n = n.
            lag: Lag size before start rolling.

        Returns:
            Series or Dataframe of the result.
        """
        m = int(self.freq / self.base_freq)
        if m == 1:
            return self._rolling(data, n, function, min_n, lag)
        else:
            e = len(data)
            res = []
            for i in range(m):
                x = data.iloc[np.arange(i, e, m, dtype=int)]
                res.append(self._rolling(x, n, function, min_n, lag))

            return pd.concat(res).reindex(data.index)

    def cumret(self, ret, period=1, lag=0):
        """Compute cumulative returns between t-period and t-lag. Eg, 12-month momentum can be obtained by setting
        period = 12, lag = 1. A negative period will generate future returns: eg, period=-1, lag=0 for one-period ahead
        return.

        Args:
            ret: Series of returns.
            period: target horizon. (+) for past returns and (-) for future returns.
            lag: return start date. output = cumulative return between t-period and t-lag.

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

    def rolling_regression(self, y, x, n, min_n=None, add_constant=True, output=None):
        """Rolling OLS accounting for frequency. When output is not None, this function is slow. Use rolling_beta()
        if possible.

        NOTE: when this function is used, the first n rows of each ID should be removed. This means that when min_n < n,
        some results can be lost. Eg, if n = 5, min_n = 3, the rows with rcount in (0, 4) will be removed, and the
        results for rcount = 3, 4 will be lost.

        Args:
            y: Series of y with index=date/id.
            x: Series or Dataframe of x with index=date/id.
            n: window size in terms of base frequency.
            min_n: minimum number of observations in the window. If observations < min_n, result is nan.
            add_constant: if True, add constant to x.
            output: list of outputs supported by RollingOLS. If None, estimate coefficients only.

        Returns:
            ndarray of list of coefficients if output=None, otherwise, ndarray of output defined by output.
        """
        m = int(self.freq / self.base_freq)

        params_only = output is None

        if add_constant:
            x = sm.add_constant(x)

        if m == 1:
            model = RollingOLS(y, x, window=n, min_nobs=min_n).fit(params_only=params_only)
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
                x_ = x.iloc[np.arange(i, e, m, dtype=int)]
                model = RollingOLS(y_, x_, window=n, min_nobs=min_n).fit(params_only=params_only)
                models.append(model)
                params.append(model.params)

            if output is None:
                return pd.concat(params).reindex(y.index)
            else:
                retvals = []
                for arg in output:
                    retval = [model.__getattribute__(arg) for model in models]
                    retvals.append(pd.concat(retval).reindex(y.index))

                return tuple(retvals)

    def rolling_beta(self, data, n, min_n=None):
        """Similar to rolling_regression. This is faster but the output is limited to coefficients, R2, and
        idiosyncratic volatility.

        Args:
            data: Dataframe with index=date/id. The first column must be y and the rest X (not including constant).
            n: window size in terms of base frequency.
            min_n: minimum number of observations in the window. If observations < min_n, result is nan.

        Returns:
            ndarray of list of coefficients, R2, and idiosyncratic volatility.
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


