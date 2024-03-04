"""Quantile portfolio construction and evaluation.

This example demonstrates how to construct quantile portfolios (and the long-short) based on a firm characteristic
and evaluate their performance.

    i) Tercile portfolios will be generated from the 12-month momentum.
    ii) Different ways of setting the transaction cost will be demonstrated.

- It is assumed that the firm characteristic has been created and saved to config.output_dir + 'crspm'.

Processing time: approx. 10 seconds.
"""

import matplotlib.pyplot as plt

from pyanomaly.globals import *
from pyanomaly.fileio import read_from_file
from pyanomaly.tcost import TransactionCost, TimeVaryingCost
from pyanomaly.analytics import *
from pyanomaly.wrdsdata import WRDS


def example3():
    set_log_path(__file__)
    start_timer('Example 3')

    # Jitted functions are slow when first called due to compile time. This example runs faster by disabling jit.
    set_config(disable_jit=True)

    ###################################
    # Set variables
    ###################################
    char = 'ret_12_1'  # 12-month momentum
    char_class = char + '_class'  # Momentum class column
    ret_col = 'exret'  # Return column
    weight_col = 'me'  # Market equity column
    split = 3  # Tercile. You can also do split = [0.3, 0.7] for 3:4:3 split.
    labels = ['H', 'M', 'L', 'H-L']  # Portfolio names: 'H-L' for long short.

    ###################################
    # Load data
    ###################################
    # Read crspm data (not raw data but the output data).
    data = read_from_file('crspm')

    # NYSE me
    data['nyse_me'] = np.where(data.exchcd == 1, data[weight_col], np.nan)

    # Delete unnecessary columns to save memory (optional).
    keep_columns(data, [char, ret_col, weight_col, 'nyse_me'])

    # Shift data except for return so that all the variables at t are as of t-1 and the return is over t-1 to t.
    data = shift(data, 1, excl_cols=[ret_col])

    # Risk-free rates.
    # Set `month_end=True` since the crspm date has also been shifted to month end.
    rf = WRDS.get_risk_free_rate(month_end=True)

    ###################################
    # Transaction costs
    ###################################
    # See `pyanomaly.tcost` module for details.
    # Try one of the followings.
    # costfcn = None  # No transaction costs.
    # costfcn = 0.003  # Transaction cost of 30 basis points.
    # costfcn = TransactionCost(buy_linear=0.002, sell_linear=0.002)  # 20 (30) bps when buying (selling).

    # Assume transaction cost varies over time and with size following Brandt et al. (2009).
    costfcn = TimeVaryingCost(data[weight_col])

    # Exclude bottom 20% based on NYSE size. This should come after setting the time varying cost
    # because `TimeVaryingCost` requires all firms' me as the input.
    data = filter(data, weight_col, (0.2, None), by='nyse_me', ginfo='date')

    ###################################
    # Make tercile portfolios
    ###################################

    # Classify data on char. The highest momentum stock will be labeled 0 and the lowest 2.
    data[char_class] = classify(data[char], split, ascending=False, ginfo='date')

    portfolios = make_quantile_portfolios(data, char_class, ret_col, weight_col, rf=rf, costfcn=costfcn, names=labels)

    ###################################
    # Evaluate the portfolios
    ###################################
    # `annualize_factor=12` will annualize the results as crspm is monthly data.
    pfperfs, pfvals = portfolios.eval(annualize_factor=12)

    # Performance metrics.
    print('\nPerformance')
    print(pfperfs)

    # Time-series of portfolio value, return, ...
    print('\nValues')
    print(pfvals)

    # Plot cumulative returns.
    pfvals['cumret'].plot()
    plt.show()

    # Plot selected portfolios.
    pfvals['cumret'][['H', 'L', 'H-L']].plot()
    plt.show()

    # Evaluate the portfolios for a sub-period and ignoring transaction costs.
    pfperfs1, pfvals1 = portfolios.eval(sdate='2001-01-01', edate='2010-12-31', annualize_factor=12,
                                        return_type='gross')
    print('\nPerformance: 2001-01 - 2010-12')
    print(pfperfs1)

    # Plot cumulative returns.
    pfvals1['cumret'][['H', 'L', 'H-L']].plot()
    plt.show()

    # To access portfolio 'H'
    pf_h = portfolios['H']

    # You can also access the returns of `eval()` as follows.
    print('\nPerformance: H')
    print(pf_h.performance)  # pf_h_perf
    print('\nValues: H')
    print(pf_h.value)  # pf_h_val

    # To see the positions (constituents)
    print('\nPositions: H')
    print(pf_h.position)

    elapsed_time('End of Example 3.')


if __name__ == '__main__':
    os.chdir('../')

    example3()
