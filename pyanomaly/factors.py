"""This module defines functions to generate factor portfolios and characteristic portfolios.

    .. autosummary::
        :nosignatures:

        prepare_data_for_factors
        make_factor_portfolio
        make_factor_portfolios
        make_all_factor_portfolios
        make_char_portfolios
"""

import pandas as pd

from pyanomaly.globals import *
# config.REPLICATE_JKP = True

from pyanomaly.analytics import *
from pyanomaly.fileio import write_to_file, read_from_file
from pyanomaly.characteristics import CRSPM, CRSPDRaw, FUNDQ, FUNDA, Merge
from pyanomaly.panel import FCPanel
import pyanomaly.datatools as dt


################################################################################
#
# Factor portfolio generation
#
################################################################################

def prepare_data_for_factors(chars, monthly=True, daily=True, sdate=None):
    """Prepare data for factor portfolio generation.

    Generate firm characteristics needed to make factors.
    The output data are used as input for :func:`make_factor_portfolios`.

    Args:
        chars: List of firm characteristics to generate.
        monthly: If True, generate firm characteristics monthly.
        daily: If True, generate firm characteristics daily.
        sdate: Start date ('yyyy-mm-dd').

    Returns:
        * mdata. ``FCPanel`` of monthly data. None if `monthly` = False.
        * ddata. ``FCPanel`` of daily data. None if `daily` = False.
    """

    elapsed_time('Generating data for factor portfolios...')

    # Process crspm.
    crspm = CRSPM(chars)
    crspm.load_data(sdate)
    crspm.filter_data()
    crspm.populate(method=None)
    crspm.update_variables()
    crspm.create_chars()

    # Process fundq.
    fundq = FUNDQ(chars)
    fundq.load_data(sdate)
    fundq.remove_duplicates()
    fundq.convert_currency()
    fundq.populate(MONTHLY, limit=3, lag=4, new_date_col='date')
    fundq.create_qitems_from_yitems()
    fundq.update_variables()
    fundq.create_chars()

    # Process funda.
    funda = FUNDA(chars)
    funda.load_data(sdate)
    funda.convert_currency()
    funda.populate(MONTHLY, limit=12, lag=4, new_date_col='date')
    funda.merge_with_fundq(fundq)
    funda.update_variables()
    funda.add_crsp_me(crspm)
    funda.create_chars()

    # Merge crspm, fundq, and funda.
    merge = Merge(chars)
    merge.preprocess(crspm, None, funda, fundq, delete_data=True)
    merge.create_chars()

    ret_col = 'exret'  # excess return
    size_col = 'me'  # market cap

    merge.remove_rawdata(['gvkey', 'rf', 'exchcd', 'primary', 'exret', size_col])

    if daily:  # Merge firm characteristics with daily returns.
        crspd = CRSPDRaw(chars)
        crspd.load_data(sdate)
        crspd.filter_data()
        crspd.update_variables()
        crspd.remove_rawdata(['ym', 'gvkey', 'rf', 'exchcd', 'primary', 'exret', size_col])

        # Shift one period forward so that all variables except exret are as of the end of t-1 and exret from t-1 to t.
        exret = crspd[ret_col]
        crspd.data = crspd.shift()
        crspd.data[ret_col] = exret

        crspd.data.dropna(subset=[ret_col, size_col], inplace=True)  # non-missing return and me
        crspd.data = crspd.data[crspd.data.primary]  # primary securities only

        crspd.merge(merge[chars], on=['ym', 'permno'], right_on=['date', 'permno'], how='left')

        crspd.remove_rawdata(['rf', 'exchcd', size_col, ret_col])
        ddata = crspd

        log('Daily data generated.')
    else:
        ddata = None

    if monthly:  # Merge firm characteristics with monthly returns.
        # One-period ahead return.

        # Shift one period forward so that all variables except exret are as of the end of t-1 and exret from t-1 to t.
        exret = merge[ret_col]
        merge.data = merge.shift()
        merge.data[ret_col] = exret

        merge.data.dropna(subset=[ret_col, size_col], inplace=True)  # non-missing return and me
        merge.data = merge.data[merge.data.primary]  # primary securities only

        merge.remove_rawdata(['rf', 'exchcd', size_col, ret_col])
        mdata = merge

        log('Monthly data generated.')
    else:
        mdata = None

    return mdata, ddata


def make_factor_portfolio(panel, ret_col, char, char_split=(0.3, 0.7), nyse=True, ascending=False, size_class=None,
                          weight_col=None):
    """Make a factor portfolio.

    The procedure is as follows.

    1. Split stocks into terciles based on the values of `char` column.

        * If `nyse` = True, cut points are determined by `char` of NYSE stocks.
        * If `ascending` is True, the first quantile contains stocks with the lowest `char` values.

    2. Make the factor portfolio.

        * If `size_class` is given,

            Factor portfolio (hml) = 1/2(Small High + Big High) − 1/2(Small Low + Big Low)

            Size portfolio (smb) = 1/3(Small High + Small Mid + Small Low) - 1/3(Big High + Big Mid + Big Low)

        * Otherwise,

            Factor portfolio (hml) = High - Low.

        * High (Low) is the first (last) quantile, i.e., the factor portfolio is (high - low) when `ascending` = False
          and (low - high) when `ascending` = True.
        * If `weight_col` is given, the factor portfolio is a `weight_col`-weighted portfolio; otherwise, it is an
          equal-weight portfolio.

    Args:
        panel: ``FCPanel`` that contains data for factor generation. It should have `char`, `ret_col`, `size_class`, and
            `weight_col` (optional) columns.
        ret_col: Return column.
        char: Firm characteristic column to make a factor portfolio from.
        char_split: Tuple of splits for terciles. (0.3, 0.7) means 3:4:3 split.
        nyse: If True, cut points are determined by `char` of NYSE stocks.
        ascending: If True, the first quantile contains stocks with the lowest `char` values.
        size_class: Size class column. If given, a factor portfolio is constructed in each size group and the
            factor portfolio is the average of them.
        weight_col: Weight column. If None, stocks are equally weighted.

    Returns:
        DataFrame of the factor and its ingredient portfolios. Index = 'date'.

        * If `size_class` is given,

            columns: ['sh', 'sm', 'sl', 'bh', 'bm', 'bl', 'hml', 'smb'].

        * Otherwise,

            columns: ['h', 'm', 'l', 'hml'].
    """

    data = panel.data

    char_class = char + '_class'  # Class (group) column.

    # Classify stocks on `char`.
    if nyse:
        char_by = char + '_by'  # Column to determine quantile breakpoints: same as char but keeps only NYSE stocks.
        data[char_by] = np.where(data.exchcd == 1, data[char], np.nan)  # Keep only NYSE stocks.
        data[char_class] = classify(data[char], char_split, ascending, ginfo=panel.get_date_group_index(),
                                    by_array=data[char_by])
    else:
        data[char_class] = classify(data[char], char_split, ascending, ginfo=panel.get_date_group_index())

    if size_class:
        # 2-D sort. factors index = date/char_class, columns = size_class.
        factors = two_dim_sort(data, char_class, size_class, ret_col, weight_col=weight_col, output_dim=2)

        # Name portfolios.
        relabel_class(factors, ['h', 'm', 'l', 'hml'], axis=0)
        relabel_class(factors, ['s', 'b', 'smb'], axis=1)
        factors = factors.unstack()
        factors.columns = factors.columns.map(''.join)

        factors['hml'] = (factors['shml'] + factors['bhml']) / 2
        factors['smb'] = (factors['smbh'] + factors['smbm'] + factors['smbl']) / 3
        drop_columns(factors, ['shml', 'bhml', 'smbh', 'smbm', 'smbl', 'smbhml'])
    else:
        factors = one_dim_sort(data, char_class, ret_col, weight_col=weight_col)

        # Name portfolios.
        relabel_class(factors,  ['h', 'm', 'l', 'hml'])
        factors = factors.unstack()
        factors.columns = factors.columns.droplevel(0)

    elapsed_time(f'Factor portfolio created: {char}')

    return factors


def make_factor_portfolios(panel, factor_groups):
    """Make factor portfolios.

    Generate factor portfolios defined in `factor_groups`, which is a "factor group" or a list of them.

    **Factor group**

    A "factor group" is a dictionary that defines a factor model and has the following structure.

    .. code-block::

        {
          'factor_name': {
              'char'(str): Firm characteristic to use to make the factor
              'ascending'(bool): If True (False), the factor portfolio is low-high (high-low). Default to False
              'char_split'(tuple): How to split stocks into three groups. Default to (0.3, 0.7)
              }
        }

    Example (Fama-French 5 factors):

    .. code-block::

        ff5 = {
            'smb_ff5': {},
            'hml': dict(char='be_me'),
            'rmw': dict(char='ope_be'),
            'cma': dict(char='at_gr1', ascending=True),
        }

    Note that

        * Any firm characteristic defined in ``FUNDA``, ``FUNDQ``, ``CRSPM``, ``CRSPD``, and ``Merge`` can be used
          to generate a factor portfolio.
        * Size factor should have an empty dict.
        * Default value items can be omitted.


    **Procedure**

        1. Split stocks into two size groups (50:50) using NYSE-size cut points.
        2. Make market factor portfolio: weighted mean excess returns of all stocks.
        3. Make factor portfolios. See :func:`make_factor_portfolio`.
        4. Make size factor portfolio: average of the size factor portfolios (smb) from each factor.

    Args:
        panel: ``FCPanel`` that contains data for factor generation, generated by :func:`prepare_data_for_factors`.
        factor_groups: (List of) factor group(s).

    Returns:
        Factor portfolio dataframe with index = 'date' and columns = ['mktrf', 'rf'] + factor names in `factor_groups`.
    """

    ret_col = 'exret'  # excess return
    size_col = 'me'  # market cap

    ########################################
    # Size split
    ########################################
    size_split = [0.5,]
    size_class = 'size_class'

    panel['nyse_me'] = np.where(panel['exchcd'] == 1, panel[size_col], np.nan)  # Keep only NYSE stocks.
    panel[size_class] = classify(panel[size_col], size_split, True, ginfo=panel.get_date_group_index(),
                                by_array=panel['nyse_me'])

    ########################################
    # Factor portfolios
    ########################################
    # Market pf.
    factors = weighted_mean(panel.data, ret_col, size_col, 'date')
    factors.rename(columns={ret_col: 'mktrf'}, inplace=True)
    factors['rf'] = panel.get_date_group().rf.first()

    if isinstance(factor_groups, dict):
        factor_groups = [factor_groups]

    for factor_group in factor_groups:
        size_factor = None
        factors['_smb_'] = 0

        for factor_name, info in factor_group.items():
            if not info:  # size factor
                size_factor = factor_name
                continue

            if factor_name in factors:  # There could be a same factor in different factor groups such as ff3 and ff5.
                continue

            char = info['char']
            ascending = info['ascending'] if 'ascending' in info else False
            char_split = info['split'] if 'split' in info else (0.3, 0.7)
            nyse = info['nyse'] if 'nyse' in info else True

            factor = make_factor_portfolio(panel, ret_col, char, char_split, nyse=nyse, ascending=ascending,
                                           size_class=size_class, weight_col=size_col)
            factors[factor_name] = factor['hml']
            factors['_smb_'] += factor['smb']

        if size_factor:
            factors[size_factor] = factors['_smb_'] / (len(factor_group) - 1)
        del factors['_smb_']

    return factors


def make_all_factor_portfolios(monthly=True, daily=True, sdate=None):
    """Make all factor portfolios.

    Currently, this function generates the following factors:

        * Fama-French 3 factors: mktrf, smb_ff, hml
        * Fama-French 5 factors: mktrf, smb_ff5, hml, rmw, cma
        * Hou-Xue-Zhang 4 factors: mktrf, smb_hxz, inv, roe
        * Stambaugh-Yuan 4 factors: mktrf, smb_sy, mgmt, perf

    The DataFrame of factors is saved to ``config.factors_monthly(daily)_fname`` in ``config.output_dir``.

    Args:
        monthly: If True, generate monthly factors.
        daily: If True, generate daily factors.
        sdate: Start date ('yyyy-mm-dd').

    See Also:
        :func:`make_factor_portfolios`
    """

    ff3 = {
        'smb_ff': {},
        'hml': dict(char='be_me'),
    }

    ff5 = {
        'smb_ff5': {},
        'hml': dict(char='be_me'),
        'rmw': dict(char='ope_be'),  # operprof
        'cma': dict(char='at_gr1', ascending=True),
    }

    hxz4 = {
        'smb_hxz': {},
        'inv': dict(char='at_gr1', ascending=True),
        'roe': dict(char='niq_be'),
    }

    sy4 = {
        'smb_sy': {},
        'mgmt': dict(char='mispricing_mgmt'),
        'perf': dict(char='mispricing_perf'),
    }

    # Factors to generate.
    factor_groups = [ff3, ff5, hxz4, sy4]

    # Firm characteristics to make factor portfolios.
    chars = []
    for factor_group in factor_groups:
        for factor_name, info in factor_group.items():
            if info and (info['char'] not in chars):
                chars.append(info['char'])

    mdata, ddata = prepare_data_for_factors(chars, monthly, daily, sdate)

    if monthly:
        elapsed_time('Making factor portfolios monthly...')
        factors_monthly = make_factor_portfolios(mdata, factor_groups)
        write_to_file(factors_monthly, config.factors_monthly_fname)

    if daily:
        elapsed_time('Making factor portfolios daily...')
        factors_daily = make_factor_portfolios(ddata, factor_groups)
        write_to_file(factors_daily, config.factors_daily_fname)


################################################################################
#
# Characteristic portfolios
#
################################################################################

def make_char_portfolios(panel, char_list, weighting):
    """Make characteristic portfolios.

    Make characteristic portfolios using the method of JKP.

        1. Split stocks into terciles (1:1:1) based on a firm characteristic. Use only NYSE stocks
           (excluding bottom 20%) to determine the cut points.
        2. Make a characteristic portfolio: First Quantile - Last Quantile.

    Args:
        panel: ``FCPanel`` that contains firm characteristics in `char_list`.
            It should also have 'exret', 'me', 'primary', and 'exchcd' columns.
        char_list: List of firm characteristics to generate.
        weighting: 'ew' (equal-weight), 'vw' (value-weight), or
            'vw_cap' (value-weight capped at 0.8 NYSE-size quantile).

    Returns:
        Characteristic portfolio DataFrame with index = 'date' and
        columns = ['group', 'char', 'ret', 'signal', 'n_firms'].

            * group: 'h', 'm', 'l', or 'hml'.
            * char: characteristic name
            * ret: characteristic portfolio return
            * signal: average characteristic value
            * n_firms: number of firms
    """

    elapsed_time(f'make_char_portfolios. weighting: {weighting}')

    ret_col = 'exret'  # excess return
    size_col = 'me'  # market cap

    # Shift one period forward so that all variables except exret are as of the end of t-1 and exret from t-1 to t.
    data = panel.shift()
    data[ret_col] = panel[ret_col]

    ########################################
    # Filtering
    ########################################
    data.dropna(subset=[ret_col, size_col], inplace=True)  # non-missing return and me
    data = data[data.primary]  # primary securities only

    ########################################
    # Weight
    ########################################
    data['nyse_me'] = np.where(data['exchcd'] == 1, data[size_col], np.nan)  # NYSE me

    if weighting == 'vw':
        weight_col = size_col
    elif weighting == 'vw_cap':
        data['me_cap'] = winsorize(data[size_col], (None, 0.2), by_array=data['nyse_me'])
        weight_col = 'me_cap'
    else:  # weighting == 'ew':
        weight_col = None

    ########################################
    # Characteristic portfolios
    ########################################
    date_gb_idx = list(data.groupby(level=0).indices.values())
    # Add identifier for the bottom 20%.
    size_split = [0.2,]
    size_class = 'size_class'
    data[size_class] = classify(data[size_col], size_split, ascending=True, ginfo=date_gb_idx,
                                 by_array=data['nyse_me'])

    split = 3
    char_by = 'char_by'
    group_col = 'group'
    labels = ['h', 'm', 'l', 'hml']

    char_portfolios = []
    for char in char_list:
        data[char_by] = data[char].where(data[size_class] > 0)  # NYSE-characteristic

        # Classify data on char.
        data[group_col] = classify(data[char], split, ascending=False, ginfo=date_gb_idx,
                                    by_array=data[char_by])

        # Make quantile portfolios.
        portfolios = one_dim_sort(data, group_col, ret_col, weight_col=weight_col)
        portfolios[['signal', 'n_firms']] = one_dim_sort(data, group_col, char, function=['mean', 'count'])

        relabel_class(portfolios, labels)
        portfolios = portfolios.rename(columns={ret_col: 'ret'})
        portfolios['char'] = char
        char_portfolios.append(portfolios.reset_index(group_col))

        elapsed_time(f'Characteristic portfolio created: {char}')

    return pd.concat(char_portfolios)[['group', 'char', 'ret', 'signal', 'n_firms']]


def test_char_portfolio():
    # Test characteristic portfolio generation.

    char_list = ['ret_1_0', 'ret_12_1']

    panel = FCPanel().load('crspm')

    hml = None
    for weighting in ['vw', 'ew', 'vw_cap']:
        char_pfs = make_char_portfolios(panel, char_list, weighting)
        if hml is None:
            hml = char_pfs[char_pfs['group'] == 'hml']
            hml = hml.rename(columns={'ret': 'ret_' + weighting})
        else:
            hml['ret_' + weighting] = char_pfs.loc[char_pfs['group'] == 'hml', 'ret']

    return hml


if __name__ == '__main__':
    os.chdir('../')

    hml = test_char_portfolio()

    # factors_monthly = read_from_file(config.factors_monthly_fname)
    #
    # ff_factors = pd.read_csv('./references/ff5_factors.csv')
    # ff_factors['date'] = (100 * ff_factors['date'] + 1).astype(str)
    # ff_factors['date'] = to_month_end(pd.to_datetime(ff_factors['date']))
    # ff_factors = ff_factors.set_index('date')
    # ff_factors = ff_factors[['Mkt-RF', 'RF', 'SMB', 'HML', 'RMW', 'CMA']]
    # ff_factors = ff_factors.rename(
    #     columns={'Mkt-RF': 'mktrf', 'RF': 'rf', 'SMB': 'smb_ff5', 'HML': 'hml', 'RMW': 'rmw', 'CMA': 'cma'})
    #
    # hxz_factors = pd.read_csv('./references/q5_factors.csv')
    # hxz_factors['date'] = (10000 * hxz_factors['year'] + 100 * hxz_factors['month'] + 1).astype(str)
    # hxz_factors['date'] = to_month_end(pd.to_datetime(hxz_factors['date']))
    # hxz_factors = hxz_factors.set_index('date')
    # hxz_factors = hxz_factors[['R_MKT', 'R_F', 'R_ME', 'R_IA', 'R_ROE', 'R_EG']]
    # hxz_factors = hxz_factors.rename(
    #     columns={'R_MKT': 'mktrf', 'R_F': 'rf', 'R_ME': 'smb_hxz', 'R_IA': 'inv', 'R_ROE': 'roe'})
    #
    # sy_factors = pd.read_csv('./references/mispricing_factors.csv')
    # sy_factors['date'] = (100 * sy_factors['YYYYMM'] + 1).astype(str)
    # sy_factors['date'] = to_month_end(pd.to_datetime(sy_factors['date']))
    # sy_factors = sy_factors.set_index('date')
    # sy_factors = sy_factors[['MKTRF', 'RF', 'SMB', 'MGMT', 'PERF']]
    # sy_factors = sy_factors.rename(
    #     columns={'MKTRF': 'mktrf', 'RF': 'rf', 'SMB': 'smb_sy', 'MGMT': 'mgmt', 'PERF': 'perf'})
    #
    #
    # compare_data(factors_monthly, ff_factors, tolerance=0.1)
    # compare_data(factors_monthly, hxz_factors, tolerance=0.1)
    # compare_data(factors_monthly, sy_factors, tolerance=0.1)

    # column                matched     corr    nan_x    nan_y
    # cma                   0.00000  0.85950       0       0
    # hml                   0.00000  0.80563       0       0
    # rmw                   0.00000  0.91763       0       0
    # smb_ff5               0.00000  0.96970       0       0
    # column                matched     corr    nan_x    nan_y
    # inv                   0.00000  0.78859       0       0
    # mktrf                 0.00000  0.99913       0       0
    # rf                    0.00154  0.96608       0       0
    # roe                   0.00316  0.81876      15       0
    # smb_hxz               0.00000  0.93257      17       0
    # column                matched     corr    nan_x    nan_y
    # mgmt                  0.09877  0.89704       0       0
    # perf                  0.11883  0.91595       0       0
    # smb_sy                0.16358  0.92592       0       0

