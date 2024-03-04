"""Sorting-based portfolio analysis.

This example demonstrates how to construct quantile portfolios and carry out 1-D or 2-D sorts.

    i) Quintile portfolios will be generated from the 12-month momentum and their mean returns and t-values will be
       computed.
    ii) 5x5 2-D sort will be conducted on momentum and size.
    iii) Firm-level cross-sectional regression will be conducted.
    iv) Factor regressions will be carried out using FF 3-factors.

- It is assumed that the firm characteristic has been created and saved to config.output_dir + 'crspm'.

See also :func:`.make_char_portfolios` for characteristic portfolio creation replicating JKP's SAS code.

Processing time: approx. 10 seconds.
"""

from pyanomaly.fileio import read_from_file
from pyanomaly.analytics import *


def example2():
    set_log_path(__file__)
    start_timer('Example 2')

    # Jitted functions are slow when first called due to compile time. This example runs faster by disabling jit.
    set_config(disable_jit=True)

    ###################################
    # Set variables
    ###################################
    char = 'ret_12_1'  # 12-month momentum
    char_class = char + '_class'  # Momentum class column
    ret_col = 'exret'  # Return column
    weight_col = 'me'  # Market equity column
    split = 5  # Qintile. You can also do split = [0.2, 0.4, 0.6, 0.8].
    labels = ['H', 2, 3, 4, 'L', 'H-L']  # Momentum portfolio names: 'H-L' for long short.

    ###################################
    # Load and prepare data
    ###################################
    # Read crspm data (not raw data but the output data).
    data = read_from_file('crspm')

    # Shift data except for return so that all the variables at t are as of t-1 and the return is over t-1 to t.
    data = shift(data, 1, excl_cols=[ret_col])

    # Drop nan returns and weights.
    data = data.dropna(subset=[ret_col, weight_col])

    # If you want to exclude bottom 20% based on NYSE size...
    log(f'crspm shape before size filtering: {data.shape}')
    data['nyse_me'] = np.where(data.exchcd == 1, data[weight_col], np.nan)  # NYSE me
    data = filter(data, weight_col, (0.2, None), ginfo='date', by='nyse_me')
    log(f'crspm shape after size filtering: {data.shape}')

    ###################################
    # 1-D sort
    ###################################
    elapsed_time('1D sort start.')

    # Classify stocks on each date on momentum. class 0: highest momentum, 5: lowest momentum.
    data[char_class] = classify(data[char], split, ascending=False, ginfo='date')

    # Make value-weighted quantile portfolios.
    # Set weight_col to None for equal-weight portfolios.
    # qpfs will have index = date/char_class and columns = [ret_col]
    qpfs = one_dim_sort(data, char_class, ret_col, weight_col=weight_col)

    # You can add more columns to the portfolios. Here, I add signal (mean of the characteristic) and
    # n_firms (number of firms).
    qpfs[['signal', 'n_firms']] = one_dim_sort(data, char_class, char, function=['mean', 'count'])

    # If you want to see the average size and price of the portfolios...
    qpfs[['size', 'prc']] = one_dim_sort(data, char_class, [weight_col, 'prc'])

    # Rename the portfolios. Without this, the names are 0, 1, ...
    relabel_class(qpfs, labels)

    # Check the result.
    print(f'Quantile portfolios sorted on {char}')
    print(qpfs)

    # Calculate the time-series mean and t-value of the returns of the quantile portfolios.
    avg, tval = time_series_average(qpfs[ret_col], cov_type='HAC', cov_kwds={'maxlags': 12})

    # Put them together to make a nice table.
    qpfs_mean = pd.concat([avg, tval], axis=1, keys=['mean', 't-val'])
    print(f'\n Mean of portfolio returns sorted on {char}')
    print(qpfs_mean)

    elapsed_time('1D sort end.')

    ###################################
    # 2-D sort
    ###################################
    elapsed_time('2D sort start.')

    # Make 5x5 portfolios on momentum and size.
    char2 = 'me'  # Market equity
    char_class2 = char2 + '_class'  # Size class column
    split2 = 5  # Quintiles
    labels2 = ['S', 2, 3, 4, 'B', 'S-B']  # Size portfolio names

    # Classify stocks on each date on size. class 0: smallest, 5: biggest.
    data[char_class2] = classify(data[char2], split2, ascending=True, ginfo='date')

    # Make 5x5 value-weighted portfolios.
    # qpfs2 will have index = date/char_class and columns = char_class2.
    qpfs2 = two_dim_sort(data, char_class, char_class2, ret_col, weight_col=weight_col, output_dim=2)

    # Rename the portfolios.
    relabel_class(qpfs2, labels)
    relabel_class(qpfs2, labels2, axis=1)

    # Calculate the time-series mean and t-value of the returns of the quantile portfolios.
    avg, tval = time_series_average(qpfs2, cov_type='HAC', cov_kwds={'maxlags': 12})
    print(f'\nMean of portfolio returns sorted on {char} and {char2}')
    print('\nMean')
    print(avg)
    print('\nt-stat')
    print(tval)

    elapsed_time('2D sort end.')

    ###################################
    # Firm-level cross-sectional regression
    ###################################
    # Cross-sectionally regress return at t on t-1 momentum and size.
    # Take the time-series average of the coefficients and examine their significance.

    elapsed_time('CS regression start.')

    y_col = ret_col
    X_cols = [char, char2]

    # Run the regressions. Output: average coefficients, t-values, and coefficient time-series.
    avg, tval, coef = crosssectional_regression(data, y_col, X_cols, add_constant=True, cov_type='HAC',
                                                cov_kwds={'maxlags': 12})

    print('\nMean of the coefficients')
    print(pd.concat([avg, tval], axis=1))
    print('\nCoefficients time-series')
    print(coef)

    elapsed_time('CS regression end.')

    ###################################
    # Factor regression
    ###################################
    # Regress quantile portfolios' returns on the FF-3 factors.

    elapsed_time('Factor regression start.')

    # Sample period.
    sdate = '1952-01'
    edate = '2020-12'

    qpfs = qpfs[sdate:edate]

    # Read factors.
    # Here, I use the factors generated by PyAnomaly. You can use different sources if you like.
    factors = read_from_file(config.factors_monthly_fname)
    # factors = WRDS.read_data('factors_monthly')  # factors from K.French website.

    # X: Constant + FF3.
    X = factors.loc[sdate:edate, ['mktrf', 'smb_ff', 'hml']].values
    # X = factors.loc[sdate:edate, ['mktrf', 'smb', 'hml']].values  # If you use K. French data.
    X = sm.add_constant(X)

    # Run the regression for each portfolio.
    result = {}
    for pf in labels:
        # y: Returns of a quantile portfolio
        y = qpfs.loc[(slice(None), pf), ret_col].values

        model = sm.OLS(y, X).fit()

        # Make an output table.
        result[pf] = pd.DataFrame({'coefs': model.params, 'tval': model.tvalues}, index=['const', 'mktrf', 'smb', 'hml'])

    result = pd.concat(result, axis=1)
    print('\nFactor regression: FF3')
    print(result)

    elapsed_time('Factor regression end.')
    elapsed_time('End of Example 2.')


if __name__ == '__main__':
    os.chdir('../')
    example2()
