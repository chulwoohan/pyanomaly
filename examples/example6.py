"""Playing with Panel

This example demonstrates how to use :class:`.Panel` class.
"""

from pyanomaly.panel import Panel
from pyanomaly.analytics import *


def example6():
    #######################################
    # Monthly data
    #######################################

    data1 = pd.DataFrame({
        'id': [100, 100, 100, 200, 200, 200, 300, 300, 300],
        'date': ['2023-03-31', '2023-04-30', '2023-05-31', '2023-03-31', '2023-04-30', '2023-05-31', '2023-03-31',
                 '2023-04-30', '2023-05-31'],
        'ret': [0.01, 0.05, 0.02, 0.03, 0.11, -0.03, np.nan, 0.03, -0.05],
        'me': [100, 120, 110, 5000, 4500, 4700, 2000, 2100, 1900],
        'me_nyse': [np.nan, np.nan, np.nan, 5000, 4500, 4700, 2000, 2100, 1900],
    })

    data1['date'] = pd.to_datetime(data1['date'])
    data1 = data1.sort_values(['id', 'date']).set_index(['date', 'id'])

    # Instantiate Panel.
    panel1 = Panel(data1, freq=MONTHLY)
    del data1
    print(panel1.data)


    # Inspect data.
    #   - 'summary': Data shape, number of unique dates, and number of unique ids.
    #   - 'id_count`: Number of ids per date.
    #   - 'nans': Number of nans and infs per column.
    #   - 'stats': Descriptive statistics. Same as ``data.describe()``.
    panel1.inspect_data(option=['summary', 'id_count', 'nans', 'stats'])


    #######################################
    # Quarterly data
    #######################################

    data2 = pd.DataFrame({
        'id': [100, 100, 100, 200, 200, 200],
        'date': ['2023-03-31', '2023-06-30', '2023-09-30', '2023-01-31', '2023-04-30', '2023-07-31'],
        'prc': [100, 120, 110, 5000, 4500, 4700],
    })

    data2['date'] = pd.to_datetime(data2['date'])
    data2 = data2.sort_values(['id', 'date']).set_index(['date', 'id'])

    # Instantiate Panel.
    panel2 = Panel(data2, freq=QUARTERLY)
    del data2
    print(panel2.data)

    # Populate data to monthly.
    panel2.populate(MONTHLY, method='ffill', limit=3)
    print(panel2.data)

    #######################################
    # Id-level operations
    #######################################
    # The methods below perform the operation per id and accounting for the data frequency.
    # For instance, if a data is a quarterly data populated monthly, ``shift(1)`` will shift the data by 3 (one quarter)
    # within each id.

    # Shift panel1.data by one month.
    data1_lag1 = panel1.shift(n=1)
    print(data1_lag1)

    # Shift panel2.data by one quarter.
    data2_lag1 = panel2.shift(n=1)
    print(data2_lag1)

    # Shift panel1.data['ret'] by 2 months.
    # The two lines below are equivalent.
    ret_lag2 = panel1.shift(panel1['ret'], 2)
    ret_lag2 = panel1.shift('ret', 2)
    print(ret_lag2)

    # Quarterly price difference.
    # The two lines below are equivalent.
    prc_diff = panel2.diff(panel2.data['prc'], 1)
    prc_diff = panel2.diff('prc', 1)
    print(prc_diff)

    # Quarterly price percentage change.
    # The two lines below are equivalent.
    prc_change = panel2.pct_change(panel2.data['prc'], 1)
    prc_change = panel2.pct_change('prc', 1)
    print(prc_change)

    # Two-period cumulative returns.
    cum_ret = panel1.cumret('ret', 2)
    print(cum_ret)

    # One-period ahead returns.
    fut_ret = panel1.futret('ret', 1)
    print(fut_ret)

    # Rolling-apply a function to each id.
    # Supported functions: 'sum', 'mean', 'std', 'var'.
    rolling_sum = panel1.rolling(panel1[['ret', 'me']], 2, 'sum', min_n=1)
    print(rolling_sum)

    # To apply a user-defined function to each id, use ``apply_to_ids()``.
    # Below is the same as the above except that the return value is ndarray.
    rolling_sum = panel1.apply_to_ids(panel1[['ret', 'me']], roll_sum, None, 2, 1)
    print(rolling_sum)

    #######################################
    # Date-level operations
    #######################################
    # To apply a user-defined function to each date, use ``apply_to_dates()``.

    # Functino to apply: sum of each column.
    def sum0(data):
        return np.sum(data, axis=0).reshape(1, -1)

    column_sum = panel1.apply_to_dates(panel1[['ret', 'me']], sum0, None)
    print(column_sum)

    # Rolling regression
    # The first column of the first argument is used as the dependent variable and the rest as the independent variables.
    # First K (number of independent varialbes) columns of the return are coefficients, the next column is R-squared,
    # and the last column is idiosyncratic volatility (standard deviation of residuals).
    retval = panel1.rolling_regression(panel1[['ret', 'me']], 3, min_n=2, add_const=True)
    retval = pd.DataFrame(retval, columns=['alpha', 'me', 'R2', 'Idio vol'], index=panel1.data.index)
    print(retval)

    #######################################
    # Panel merge
    #######################################
    # Two panels can be merged as follows.
    panel1.merge(panel2, on=['date', 'id'], how='left')
    print(panel1.data)


if __name__ == '__main__':
    os.chdir('../')

    example6()

