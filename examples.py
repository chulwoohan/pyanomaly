import matplotlib.pyplot as plt

import pyanomaly as pa
from pyanomaly.characteristics import *
from pyanomaly.jkp import make_factor_portfolios
from pyanomaly.tcost import TransactionCost, TimeVaryingCost

"""
EXAMPLE 1: Firm characteristics generation (Replication of JKP).

This example shows the full process of i) data download, ii) factor creation, and iii) characteristics creation.
This example replicates the JKP's SAS code as closely as possible.

In this example, 
    i) funda is merged with fundq to create quarterly updated annual accounting data.
    ii) The latest market equity (me) is used when creating firm characteristics.
    iii) Generate firm characteristics that appear in the JKP's paper.
    
Total processing time: approx. 1 hour.
"""

def example1():
    # Set log file path. Without this, the log will be printed in stdout.
    set_log_path('./log/example1.log')
    # initialize time check.
    elapsed_time('Start of Example 1.')

    # Generate characteristics in 'jkp' column in mapping.xlsx.
    alias = 'jkp'
    # Start date. Set to None to create characteristics from as early as possible.
    sdate = '1950-01-01'

    ###################################
    # Data download
    ###################################
    drawline()  # This will print a line in the log file. Only for visual effect.
    log('DOWNLOADING DATA')
    drawline()

    wrds = WRDS('wrds_username')  # Use your WRDS user id.
    # Download all necessary data.
    wrds.download_all()
    # Create crspm(d) from m(d)sf and m(d)seall and add gvkey to them.
    wrds.preprocess_crsp()

    ###################################
    # Factor generation
    ###################################
    # Make FF3 and HXZ4 factors. These are used in some firm characteristics, such as the residual momentum.
    drawline()
    log('FACTOR PORTFOLIOS')
    drawline()

    make_factor_portfolios(monthly=True)  # Monthly factors
    make_factor_portfolios(monthly=False)  # Daily factors

    ###################################
    # CRSPM
    ###################################
    # Generate firm characteristics from crspm.
    drawline()
    log('PROCESSING CRSPM')
    drawline()

    crspm = CRSPM(alias=alias)
    # Load crspm.
    crspm.load_data(sdate)

    # Filter data on shrcd, ...
    crspm.filter_data()
    # Fill missing months by populating the data.
    # There are only few missing data and the results aren't affected much by this.
    crspm.populate(freq=MONTHLY, method=None)
    # Some preprocessing, e.g., creating frequently used variables.
    crspm.update_variables()
    # Merge crspm with the factors created earlier.
    crspm.merge_with_factors()

    # Display what characteristics will be generated. Just for information.
    crspm.show_available_functions()
    # Create characteristics.
    crspm.create_chars()

    # Postprocessing: delete temporary variables, etc.
    crspm.postprocess()
    # Saves the results. You can give a file name if you wish. Otherwise, the file name will be the lower-case class
    # name, i.e., crspm. The file can later be loaded using the method, crspm.load().
    crspm.save()

    ###################################
    # CRSPD
    ###################################
    # Generate firm characteristics from crspd.
    drawline()
    log('PROCESSING CRSPD')
    drawline()

    crspd = CRSPD(alias=alias)
    # Load crspd.
    crspd.load_data(sdate)

    # Filter data on shrcd, ...
    crspd.filter_data()
    crspd.update_variables()
    crspd.merge_with_factors()

    crspd.show_available_functions()
    crspd.create_chars()

    crspd.postprocess()
    crspd.save()

    ###################################
    # FUNDQ
    ###################################
    # Generate firm characteristics from fundq.
    drawline()
    log('PROCESSING FUNDQ')
    drawline()

    fundq = FUNDQ(alias=alias)
    # Load fundq.
    fundq.load_data(sdate)

    # fundq has some duplicates (same datedate/gvkey). Drop duplicates.
    fundq.remove_duplicates()
    # Convert values in another currency (currently only CAD) to USD values.
    fundq.convert_currency()
    # Populate data to monthly.
    fundq.convert_to_monthly()
    # Make quarterly variables from ytd variables and use them to fill missing quarterly variables.
    fundq.create_qitems_from_yitems()
    fundq.update_variables()

    fundq.show_available_functions()
    fundq.create_chars()

    fundq.postprocess()
    fundq.save()

    ###################################
    # FUNDA
    ###################################
    # Generate firm characteristics from funda.
    drawline()
    log('PROCESSING FUNDA')
    drawline()

    funda = FUNDA(alias=alias)
    # Load fundq.
    funda.load_data(sdate)

    funda.convert_currency()
    funda.convert_to_monthly()
    # Generate quarterly-updated funda data from fundq and merge them with funda.
    funda.merge_with_fundq(fundq)
    funda.update_variables()
    # Add the market equity of crspm to funda.
    funda.add_crsp_me(crspm)

    funda.show_available_functions()
    funda.create_chars()

    funda.postprocess()
    funda.save()

    ###################################
    # Merge
    ###################################
    # Combine everything together and generate firm characteristics that require data from multiple data sources.
    drawline()
    log('PROCESSING MERGE')
    drawline()

    merge = Merge()
    # Merge all data together.
    merge.preprocess(crspm, crspd, funda, fundq)

    merge.show_available_functions()
    merge_chars = merge.get_available_chars()
    merge.create_chars(merge_chars)

    merge.postprocess()
    # By default, `Panel.save()` saves all the columns of `Panel.data`.
    # If you want to save only the variables you need to save the disc space, you can do, e.g.:
    # columns = ['gvkey', 'datadate', 'primary', 'me', 'ret', 'exret', 'rf']
    # merge.save(other_columns=columns)
    # Then, all firm characteristics plus the columns in `columns` will be saved.
    merge.save()

    elapsed_time('End of Example 1.')


"""
EXAMPLE 2: Different ways of generating characteristics.

This example demonstrates 
    i) how to generate a few selected firm characteristics without adding a column to mapping.xlsx;
    ii) how to set the output file path instead of the default path.

- It is assumed that data has been downloaded from WRDS.
- Short-term reversal and 12M-momentum will be generated.
- 'alias' is set to None, which means function names are used as the aliases. 

Process time: approx. 20 seconds.
"""

def example2():
    elapsed_time('Start of Example 2.')

    alias = None  # alias = function names
    sdate = None  # create characteristics from as early as possible.
    # Characteristics to generate: short-term reversal and 12M-momentum.
    chars = ['ret_1_0', 'ret_12_1']

    # Generate firm characteristics from crspm.
    crspm = CRSPM(alias=alias)
    crspm.load_data(sdate)

    # Filter data on shrcd, ...
    crspm.filter_data()
    # Let's skip populate() to save time as it doesn't change the result much.
    # crspm.populate(freq=MONTHLY, method=None)

    # Some preprocessing, e.g., creating frequently used variables.
    crspm.update_variables()

    # Give a list of characteristics to generate.
    crspm.create_chars(chars)

    crspm.postprocess()

    # Save the output to ./output2/example2.pickle.
    # `other_columns=[]` will save only the characterisitcs.
    crspm.save('example2', fdir='output2', other_columns=[])

    elapsed_time('End of Example 2.')


"""
EXAMPLE 3

This example demonstrates how to generate funda characteristics using only funda data, i.e., 
i) without merging with annualized fundq data and ii) using market equity from funda (funda.prcc_f * funda.csho).

- It is assumed that data has been downloaded from WRDS.
- funda and fundq characteristics defined in 'jkp' column are generated.
- It is assumed that funda data is available 6 months later and fundq 4 months later.
- Choose only the stocks denominated in USD: currency conversion is unnecessary.
- The output file will contain the characteristics as well as the raw data from funda and fundq.
"""

def example3():
    elapsed_time('Start of Example 3.')

    alias = 'jkp'  # Generate characteristics defined in 'jkp' column.
    sdate = None  # create characteristics from as early as possible.

    drawline()
    log('PROCESSING FUNDQ')
    drawline()
    fundq = FUNDQ(alias=alias)
    fundq.load_data(sdate)
    fundq.preprocess()
    fundq.create_chars()
    fundq.postprocess()

    # Generate firm characteristics from funda.
    drawline()
    log('PROCESSING FUNDA')
    drawline()
    funda = FUNDA(alias=alias)
    funda.load_data(sdate)
    funda._filter_data(('curcd', '==', 'USD'))  # USD stocks only. This is equivalent to
                                                # funda.data = funda.data[funda.data['curcd'] == 'USD']
    funda.convert_to_monthly(lag=6)  # funda data available 6 months later.
    funda.update_variables()
    funda.create_chars()
    funda.postprocess()

    funda.merge(fundq, how='left')  # Left-join funda with fundq on index (date/gvkey).
    funda.save('funda_fundq_jkp')  # Output file = output/funda_fundq_jkp.pickle.

    elapsed_time('End of Example 3.')


"""
EXAMPLE 4: Defining a new characteristic.

This example demonstrates how to define a new class by inheriting CRSPM and add a new characteristic.

A new characteristic, 'excess_ret_change', will be added, which is defined as the excess return over the market return 
divided by the one-year average excess return (I have no idea if this has any predictive power).

- It is assumed that data has been downloaded from WRDS.
"""

# Define a class by inheriting CRSPM.
class MyCRSPM(CRSPM):

    # Function for the new characteristic.
    # Note that function name should be of the form 'c_[characteristic name]'.
    def c_excess_ret_change(self):
        """Change of excess return over the market return."""
        # Make a short (one-line) docstring for description.
        # This is displayed when `show_available_functions()` is called.

        cm = self.data

        # One-year average excess return
        avg_exret = self.rolling(cm.exret - cm.mktrf, 12, 'mean')
        # Excess return.
        exret = cm.exret - cm.mktrf

        # Characteristic
        char = exret / avg_exret

        return char

def example4():
    elapsed_time('Start of Example 4.')

    alias = None  # alias = function names
    sdate = None  # create characteristics from as early as possible.
    # Characteristics to generate: the function name is the characteristic name without alias.
    chars = ['excess_ret_change']

    # Instantiate the class defined above.
    crspm = MyCRSPM(alias=alias)
    crspm.load_data(sdate)

    crspm.filter_data()
    crspm.update_variables()

    # We need to add factors as we need the market return.
    crspm.merge_with_factors()

    # Let's see if 'excess_ret_change' has been added.
    crspm.show_available_functions()

    # Let's create the characteristic!
    crspm.create_chars(chars)

    crspm.postprocess()
    crspm.save('example4')

    # Later, you can load the saved file using 'read_from_data()'
    data = read_from_file('example4')
    print(data['excess_ret_change'])

    elapsed_time('End of Example 4.')


"""
EXAMPLE 5: Sorting-based portfolio analysis.

This example demonstrates how to construct quantile portfolios and carry out 1-D or 2-D sorts.
    i) Quintile portfolios will be generated from the 12-month momentum and their mean returns and t-values will be 
       computed.
    ii) 5x5 2-D sort will be conducted on momentum and size. 
    iii) Firm-level cross-sectional regression will be conducted.
    iv) Factor regressions will be carried out using FF 3-factors.

- It is assumed that the firm characteristic has been created and saved to 'crspm.pickle'.

Process time: approx. 24 seconds.
"""

def example5():
    elapsed_time('Start of Example 5.')

    ###################################
    # Set variables
    ###################################
    char = 'ret_12_1'  # 12-month momentum
    char_class = char + '_class'  # Group column: portfolio-stock mapping
    ret_col = 'futret'  # Future return column
    weight_col = 'me'  # Market equity column
    split = 5  # Qintile. You can also do split = [0.2, 0.4, 0.6, 0.8].
    labels = ['H', 2, 3, 4, 'L', 'H-L']  # Portfolio names: 'H-L' for long short.

    ###################################
    # Load data
    ###################################
    # Read crspm data (not raw data but the output data).
    data = read_from_file('crspm')

    # If you want to exclude bottom 20% based on NYSE size...
    data['nyse_me'] = np.where(data.exchcd == 1, data[weight_col], np.nan)  # NYSE me
    data = filter(data, weight_col, (0.2, None), by='nyse_me')

    # Make the future return. If it's already in the data, this step can be skipped.
    data[ret_col] = make_future_return(data['exret'])  # Excess return

    # Note that the future return at t is the return between t and t+1 and other values are as of t.
    # If you want the return at t to be the return between t-1 and t and all other values as of t-1,
    # you have to shift forward data as follows.
    data = data.groupby(level=-1).shift(1)  # Shift data forward.
    data = data.dropna(subset=[ret_col])  # Drop nan returns.

    ###################################
    # 1-D sort
    ###################################
    # Classify stocks based on char.
    data[char_class] = classify(data[char], split, ascending=False)

    # Make quantile portfolios.
    # qpfs will have index = date/classes and columns = ['futret']
    qpfs = one_dim_sort(data, char_class, ret_col, weight_col=weight_col)

    # You can add more columns to the portfolios. Here I add `signal` (mean of the characteristic) and
    # n_firms (number of firms).
    qpfs[['signal', 'n_firms']] = one_dim_sort(data, char_class, char, function=['mean', 'count'])

    # If you want to see the average size and price of the portfolios...
    qpfs[['size', 'prc']] = one_dim_sort(data, char_class, [weight_col, 'prc'])

    # Name the portfolios. Without this, the names are 0, 1, ...
    relabel_class(qpfs, labels)

    print(f'Quantile portfolios sorted on {char}')
    print(qpfs)

    # Calculate the time-series mean and t-value of the future return for each portfolio.
    # avg and tval have index values equal to the portfolio names.
    avg, tval = time_series_average(qpfs[ret_col], cov_type='HAC', cov_kwds={'maxlags': 12})
    print(f'\n Mean of portfolio returns sorted on {char}')
    print(avg)
    print(tval)
    # Put them together to make a nice table.
    print(pd.concat([avg, tval], axis=1, keys=['mean', 't-val']))

    ###################################
    # 2-D sort
    ###################################
    # Let's make a 5x5 portfolios on char and char2.
    char2 = 'me'
    char_class2 = char2 + '_class'
    split2 = 5
    labels2 = ['S', 2, 3, 4, 'B', 'S-B']

    # Classify stocks based on char2 in ascending order.
    data[char_class2] = classify(data[char2], split2, ascending=True)

    # Make 5x5 portfolios.
    # qpfs2 will have index = date/char_class/char_class2 and columns = ['futret'].
    qpfs2 = two_dim_sort(data, char_class, char_class2, ret_col, weight_col=weight_col)

    # Make char_class2 as columns. Now columns are ['S', 2, 3, 4, 'B', 'S-B']
    qpfs2 = qpfs2.unstack()
    qpfs2.columns = qpfs2.columns.droplevel(0)

    # Name the portfolios. We relabel after unstacking as `unstack()` sort values.
    relabel_class(qpfs2, labels)
    relabel_class(qpfs2, labels2, axis=1)

    # Calculate the time-series mean and t-value of the future return for each portfolio.
    avg, tval = time_series_average(qpfs2, cov_type='HAC', cov_kwds={'maxlags': 12})
    print(f'\n Mean of portfolio returns sorted on {char} and {char2}')
    print(avg)
    print(tval)

    ###################################
    # Firm-level cross-sectional regression.
    ###################################
    # Regress return cross-sectionally on the t-1 characteristic and other variables at each date t.
    # Take the time-series average of the coefficients and examine their significance.

    # X: For simplicity, we will only use the characteristic and the size.
    exog_cols = [char, 'me']
    # Run the regressions. This function returns the average coefficients, t-values, and coefficients time-series.
    avg, tval, coef = crosssectional_regression(data, ret_col, exog_cols, add_constant=True, cov_type='HAC',
                                                cov_kwds={'maxlags': 12})
    print('\nMean of the coefficients')
    print(pd.concat([avg, tval], axis=1))
    print('\nCoefficients time-series')
    print(coef)

    ###################################
    # Factor regression
    ###################################
    # Regress the long-short portfolio (H-L) returns on the FF-3 factors.

    # Sample period.
    sdate = '1952-01'
    edate = '2020-12'

    # Read factors.
    # Here I use the factors generated by PyAnomaly. You can use different sources if you like.
    factors = read_from_file(config.factors_monthly_fname)
    # factors = WRDS.read_data('factors_monthly')  # factors from K.French website.

    qpfs = qpfs[sdate:edate]
    factors = factors[sdate:edate]

    # X: Constant + FF3.
    X = factors.loc[sdate:edate, ['mktrf', 'smb_ff', 'hml']].values
    # X = factors.loc[sdate:edate, ['mktrf', 'smb', 'hml']].values  # If you use K. French data.
    X = sm.add_constant(X)

    # Run the regression for each portfolio.
    result = {}
    for pf in qpfs.index.get_level_values(-1):
        # y: Returns of a portfolio
        y = qpfs.loc[(slice(None), pf), 'futret'].values

        model = sm.OLS(y, X).fit()

        # Make an output table.
        result[pf] = pd.DataFrame({'coefs': model.params, 'tval': model.tvalues}, index=['const', 'mktrf', 'smb', 'hml'])

    result = pd.concat(result, axis=1)
    print('\nFactor regression: FF3')
    print(result)

    elapsed_time('Start of Example 5.')


"""
EXAMPLE 6: Quantile portfolio construction and performance evaluation.

This example demonstrates how to construct quantile portfolios (and the long-short) based on a firm characteristic 
and evaluate their performance.
    i) Tercile portfolios will be generated from the 12-month momentum.
    ii) Different ways of setting the transaction cost will shown.

- It is assumed that the firm characteristic has been created and saved to 'crspm.pickle'.

See also `pyanomaly.jkp` module for factor and characteristic portfolio creation replicating JKP's SAS code.

Process time: approx. 10 seconds.
"""

def example6():
    elapsed_time('Start of Example 6.')

    ###################################
    # Set variables
    ###################################
    char = 'ret_12_1'  # 12-month momentum
    char_class = char + '_class'  # Group column: portfolio-stock mapping
    ret_col = 'futret'  # Future return column
    weight_col = 'me'  # Market equity column
    split = 3  # Tercile. You can also do split = [0.2, 0.8] for 2:6:2 split.
    labels = ['H', 'M', 'L', 'H-L']  # Portfolio names: 'H-L' for long short.

    ###################################
    # Load data
    ###################################
    # Read crspm data (not raw data but the output data).
    data = read_from_file('crspm')

    # If you want to exclude bottom 20% based on NYSE size...
    data['nyse_me'] = np.where(data.exchcd == 1, data[weight_col], np.nan)  # NYSE me
    data = filter(data, weight_col, (0.2, None), by='nyse_me')

    # Make the future return. If it's already in the data, this step can be skipped.
    data[ret_col] = make_future_return(data['ret'])

    # Let's just keep the data we need. Below is the same as `data = data[[char, ret_col, weight_col]]`
    # but faster and more memroy efficient.
    keep_columns(data, [char, ret_col, weight_col])

    # Risk-free rates.
    # Don't forget to set `month_end=True` since the crspm date has also been shifted to month end.
    rf = WRDS.get_risk_free_rate(month_end=True)

    ###################################
    # Transaction costs
    ###################################
    # See `pyanomaly.tcost` module for details.
    # Try one of the followings.
    # costfcn = None  # No transaction costs.
    # costfcn = 0.003  # Transaction cost of 30 basis points.
    # costfcn = TransactionCost(buy_linear=0.002, sell_linear=0.002)  # 20 (30) bps when buying (selling).

    # In this example, we will assume transaction cost that decreases over time and with size.
    costfcn = TimeVaryingCost(data[weight_col])

    ###################################
    # Make portfolios
    ###################################
    # Classify data on char. The highest momentum stock will be labeled 0 and the lowest 2.
    data[char_class] = classify(data[char], split, ascending=False)

    # Make position data.
    # `make_position()` converts data to position data that is used as input to `Portfolio` class.
    # In `data`, the future return at t is the return between t and t+1, whereas in the position data, dates are shifted
    # so that the future return at t is the return between t-1 and t.
    # Set weight_col = None for equally-weighted portfolios.
    # `other_cols` are the columns you want to keep in the output.
    position = make_position(data, ret_col, weight_col, char_class, other_cols=None)

    # Make portfolios. `portfolios` will have four portfolios, 'H', 'M', 'L', and 'H-L'.
    portfolios = make_quantile_portfolios(position, char_class, rf=rf, costfcn=costfcn, labels=labels)

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

    # Let's plot cumulative returns.
    pfvals['cumret'].plot()
    plt.show()

    # To plot selected portfolios...
    pfvals['cumret'][['H', 'L', 'H-L']].plot()
    plt.show()

    # Evaluate the portfolio for a sub-period ignoring transaction costs.
    pfperfs1, pfvals1 = portfolios.eval(sdate='2001-01-01', edate='2010-12-31', annualize_factor=12, consider_cost=False)
    print('\nPerformance: 2001-01 - 2010-12')
    print(pfperfs1)
    pfvals1['cumret'][['H', 'L', 'H-L']].plot()
    plt.show()

    # To access portfolio 'H'
    pf_h = portfolios['H']
    pf_h_perf, pf_h_val = pf_h.eval()  # no annualize

    # You can also access the returns of `eval()` as follows.
    print('\nPerformance: H')
    print(pf_h.performance)  # pf_h_perf
    print('\nValues: H')
    print(pf_h.value)  # pf_h_val

    # To see the positions (constituents)
    print('\nPositions: H')
    print(pf_h.position)

    elapsed_time('End of Example 6.')


"""
EXAMPLE 7: Table download

This example demonstrates how to download a new table from WRDS.
Different ways of downloading comp.secm table will be demonstrated.
"""

def example7():
    elapsed_time('Start of Example 7.')

    wrds = WRDS('fehouse')

    # Download the entire table at once.
    # wrds.download_table('comp', 'secm', date_cols=['datadate'])  # 'datadate's type will be converted to datetime.

    # Download the entire table asynchronously.
    # This downloads data every `interval` years. For a small size data, this can be slower than `download_table()`.
    wrds.download_table_async('comp', 'secm', date_col='datadate', date_cols=['datadate'], interval=5)

    # When you need only some fields, they can be given in `sql` as follows.
    sql = 'datadate, gvkey, cshoq, prccm'
    wrds.download_table_async('comp', 'secm', sql=sql, date_col='datadate', date_cols=['datadate'])

    # You can also download data using a complete query statement.
    # Below is equivalent to the above.
    # Note that the query statement must contain 'WHERE [`date_col`] BETWEEN {} and {}'.
    sql = f"""
        SELECT datadate, gvkey, cshoq, prccm
        FROM comp.secm
        WHERE datadate between '{{}}' and '{{}}'
    """
    wrds.download_table_async('comp', 'secm', sql=sql, date_cols=['datadate'])

    elapsed_time('End of Example 7.')


if __name__ == '__main__':
    example5()

    # # Set log file path. Without this, the log will be printed in stdout.
    # set_log_path('./log/example1.log')
    # # initialize time check.
    # elapsed_time('Start of Example 1.')
    #
    # # Generate characteristics in 'jkp' column in mapping.xlsx.
    # alias = 'jkp'
    # # Start date. Set to None to create characteristics from as early as possible.
    # sdate = None
    #
    # # crspm = CRSPM()
    # # crspm.load()
    # #
    # # crspd = CRSPD()
    # # crspd.load()
    # #
    # # fundq = FUNDQ()
    # # fundq.load()
    # #
    # # funda = FUNDA()
    # # funda.load()
    #
    # ###################################
    # # CRSPM
    # ###################################
    # # Generate firm characteristics from crspm.
    # drawline()
    # log('PROCESSING CRSPM')
    # drawline()
    #
    # crspm = CRSPM(alias=alias)
    # # Load crspm.
    # crspm.load_data(sdate)
    #
    # # Filter data on shrcd, ...
    # crspm.filter_data()
    # # Fill missing months bu populating the data.
    # # There are only few missing data and the results aren't affected much by this.
    # crspm.populate(freq=MONTHLY, method=None)
    # # Some preprocessing, e.g., creating frequently used variables.
    # crspm.update_variables()
    # # Merge crspm with the factors created earlier.
    # crspm.merge_with_factors()
    #
    # # Display what characteristics will be generated. Just for information.
    # crspm.show_available_functions()
    # # Create characteristics.
    # crspm.create_chars()
    #
    # # Postprocessing: delete temporary variables, etc.
    # crspm.postprocess()
    # # Saves the results. You can give a file name if you wish. Otherwise, the file name will be the lower-case class
    # # name, i.e., crspm. The file can later be loaded using the method, crspm.load_data().
    # crspm.save()
    #
    # ###################################
    # # CRSPD
    # ###################################
    # # Generate firm characteristics from crspd.
    # drawline()
    # log('PROCESSING CRSPD')
    # drawline()
    #
    # crspd = CRSPD(alias=alias)
    # # Load crspd.
    # crspd.load_data(sdate)
    #
    # # Filter data on shrcd, ...
    # crspd.filter_data()
    # crspd.update_variables()
    # crspd.merge_with_factors()
    #
    # crspd.show_available_functions()
    # crspd.create_chars()
    #
    # crspd.postprocess()
    # crspd.save()
    #
    # ###################################
    # # FUNDQ
    # ###################################
    # # Generate firm characteristics from fundq.
    # drawline()
    # log('PROCESSING FUNDQ')
    # drawline()
    #
    # fundq = FUNDQ(alias=alias)
    # # Load fundq.
    # fundq.load_data(sdate)
    #
    # # fundq has some duplicates (same datedate/gvkey). Drop duplicates.
    # fundq.remove_duplicates()
    # # Convert values in another currency (currently only CAD) to USD values.
    # fundq.convert_currency()
    # # Populate data to monthly.
    # fundq.convert_to_monthly()
    # # Make quarterly variables from ytd variables and use them to fill missing quarterly variables.
    # fundq.create_qitems_from_yitems()
    # fundq.update_variables()
    #
    # fundq.show_available_functions()
    # fundq.create_chars()
    #
    # fundq.postprocess()
    # fundq.save()
    #
    # ###################################
    # # FUNDA
    # ###################################
    # # Generate firm characteristics from funda.
    # drawline()
    # log('PROCESSING FUNDA')
    # drawline()
    #
    # funda = FUNDA(alias=alias)
    # # Load fundq.
    # funda.load_data(sdate)
    #
    # funda.convert_currency()
    # funda.convert_to_monthly()
    # # Generate quarterly-updated funda data from fundq and merge them with funda.
    # funda.merge_with_fundq(fundq)
    # funda.update_variables()
    # # Add the market equity of crspm to funda.
    # funda.add_crsp_me(crspm)
    #
    # funda.show_available_functions()
    # funda.create_chars()
    #
    # funda.postprocess()
    # funda.save()
    #
    #
    # merge = Merge()
    # # Merge all data together.
    # merge.preprocess(crspm, crspd, funda, fundq)
    #
    # merge.show_available_functions()
    # merge.get_available_chars()
    # merge.create_chars()
    #
    # merge.postprocess()
    #
    # columns = ['permco', 'gvkey', 'datadate', 'primary', 'me', 'me_company', 'ret', 'exret', 'rf']
    # # merge.save('merge_jkp', other_columns=columns)
    # merge.save()
    #