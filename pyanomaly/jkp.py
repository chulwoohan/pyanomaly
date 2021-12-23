import numpy as np
import pandas as pd

from pyanomaly.globals import *
from pyanomaly.analytics import *
from pyanomaly.fileio import write_to_file
from pyanomaly.characteristics import CRSPM, CRSPD, FUNDQ, FUNDA, merge_crsp_comp


################################################################################
#
# FF and HXZ factor portfolios (JKP version)
#
################################################################################
def prepare_data_for_factors(monthly=True, sdate=None):
    """Create characteristics needed to make factors.

    Args:
        monthly: If True (False), make monthly (daily) factors.
        sdate: start date ('yyyy-mm-dd').
    """

    crsp = CRSPM() if monthly else CRSPD()
    crsp.load_data(sdate)
    crsp.preprocess()
    crsp.data['target'] = crsp.cumret(crsp.data.exret, period=-1)
    crsp.data = crsp.data[['gvkey', 'target', 'rf', 'me', 'exchcd', 'primary', 'me_company']]

    fundq = FUNDQ()
    fundq.load_data(sdate)
    fundq.remove_duplicates()
    fundq.fill_missing_qtr()
    fundq.preprocess()
    fundq.create_chars(['niq_be'])

    funda = FUNDA()
    funda.load_data(sdate)
    # funda.fill_missing_year()
    funda.merge_with_fundq(fundq)
    # funda.fill_missing_qtr()
    funda.preprocess()
    funda.add_crsp_me(crsp)
    funda.create_chars(['at_gr1', 'be_me'])

    merged_data = merge_crsp_comp(crsp.data, fundq.data[['niq_be']], gap=4, tolerance=12)
    merged_data = merge_crsp_comp(merged_data, funda.data[['at_gr1', 'be_me']], gap=4, tolerance=12)

    return merged_data


def make_factor_portfolio_(data, char, target, size_class, weight_col=None):
    """Make a factor portfolio.

    Args:
        data: Dataframe with date/id index: its columns should contain char, target, size_class, and
        weight_col (optional).
        char: characteristic column to make a factor portfolio.
        target: target (future) return column.
        size_class: column for size classes: 's', 'b'.
        weight_col: weight column. If None, stocks are equally weighted.

    Returns:
        Dataframe of (size x char) factor portfolios: index=date, columns: bh, bl, bm, sh, sl, sm, hml, smb. The column
        names can be different if labels are set differently.
    """

    char_split = [0.3, 0.7, 1.0]
    char_labels = ['h', 'm', 'l']
    char_by = char + '_by'  # column to determine quantile thresholds: same as char but keeps only NYSE stocks.
    char_class = char + '_class'  # characteristic class column name.

    data[char_by] = data[char]
    data.loc[data.exchcd != 1, char_by] = None  # Keep only NYSE stocks.

    data[char_class] = classify(data, char, char_split, ascending=False, by=char_by, labels=char_labels)

    # Remove unclassified rows.
    # data = data[(data[size_class].isin(['s', 'b'])) & data[char_class].isin(char_labels)]
    data = data[~data[size_class].isin(['', None]) & ~data[char_class].isin(['', None])]
    factors = two_dim_sort(data, size_class, char_class, target, weight_col=weight_col)

    factors = factors.reset_index(level=[-2, -1])  # index: date/size/bm -> date
    factors['class'] = factors[size_class] + factors[char_class]
    factors = factors.pivot(columns='class', values=target)

    # smb and hml factors
    factors['hml'] = (factors['bh'] + factors['sh']) / 2 - \
                     (factors['bl'] + factors['sl']) / 2
    factors['smb'] = (factors['sh'] + factors['sm'] + factors['sl']) / 3 - \
                     (factors['bh'] + factors['bm'] + factors['bl']) / 3

    elapsed_time(f'factor portfolio created: {char}')

    factors = factors.shift(1)  # Since target is the future return, factor portfolio return at t is the return
                                # from t to t+1. Shift one period so that the return is from t-1 to t.
    return factors


def make_factor_portfolios_(data):
    """Make factor portfolios. Factors generated are
        i) FF 3 factors;
        ii) HXZ 4 factors.

    Returns:
        Factor portfolio dataframe: index=data, columns = mktrf, smb_ff, hml, inv, roe, smb_hxz.
        hml, inv, roe: Factor portfolio dataframes including the component portfolios. These are returns from
        make_factor_portfolio_().
    """
    elapsed_time('make_factor_portfolios')

    target = 'target'  # t+1 excess return
    size_col = 'me'  # security level me as market equity

    ########################################
    # Filtering
    ########################################
    data = data[data.primary]  # primary securities only
    data = data.dropna(subset=[target, size_col])  # non-missing t+1 return and me

    ########################################
    # Size split
    ########################################
    size_split = [0.5, 1.0]
    size_labels = ['s', 'b']
    size_by = 'me_by'
    size_class = 'size_class'

    data[size_by] = data[size_col]
    data.loc[data.exchcd != 1, size_by] = None  # Split by NYSE me
    # data[size_by] = data.loc[data.exchcd == 1, size_col]  # Split by NYSE me (this is slower)

    data[size_class] = classify(data, size_col, size_split, ascending=True, by=size_by, labels=size_labels)
    elapsed_time('size_class generated.')

    ########################################
    # Factor portfolios
    ########################################
    # market pf.
    factors = weighted_mean(data, target, size_col, 'date')
    factors.rename(columns={target: 'mktrf'}, inplace=True)
    factors = factors.shift(1)  # shift return period from (t, t+1) to (t-1, t).
    factors['rf'] = data.groupby('date').rf.first()

    # FF factors
    hml = make_factor_portfolio_(data, 'be_me', target, size_class, size_col)
    factors[['smb_ff', 'hml']] = hml[['smb', 'hml']]

    # HXZ factors
    inv = make_factor_portfolio_(data, 'at_gr1', target, size_class, size_col)
    roe = make_factor_portfolio_(data, 'niq_be', target, size_class, size_col)
    factors['inv'] = -inv['hml']
    factors['roe'] = roe['hml']
    factors['smb_hxz'] = (inv['smb'] + roe['smb']) / 2

    return factors, hml, inv, roe


def make_factor_portfolios(monthly=True, sdate=None):
    data = prepare_data_for_factors(monthly, sdate)
    factors, hml, inv, roe = make_factor_portfolios_(data)

    fname = config.monthly_factors_fname if monthly else config.daily_factors_fname
    write_to_file(factors, fname)

    return factors, hml, inv, roe



################################################################################
#
# Characteristic portfolios
#
################################################################################
def make_char_portfolios(data, char_list, weighting):
    elapsed_time('make_char_portfolios')

    target = 'target'  # t+1 excess return
    size_col = 'me'  # security level me as market equity

    ########################################
    # Filtering
    ########################################
    data = data[data.primary]  # primary securities only
    data = data.dropna(subset=[target, size_col])  # non-missing t+1 return and me

    ########################################
    # Size split
    ########################################
    size_split = [0.2, 0.8, 1.0]
    size_by = 'me_by'
    size_class = 'size_class'

    data[size_by] = data[size_col]
    data.loc[data.exchcd != 1, size_by] = None  # Split by NYSE me
    data[size_class] = classify(data, size_col, size_split, ascending=True, by=size_by)
    elapsed_time('size_class generated.')

    if weighting == 'vw_cap':
        data['me_cap'] = data[size_col]
        data.loc[data[size_class] == 2, 'me_cap'] = data.loc[data[size_class] == 2, 'me_cap'].min()  # cap at NYSE80

    ########################################
    # Characteristic portfolios
    ########################################
    split = 3
    char_by = 'char_by'
    char_class = 'char_class'
    # data_ = data.copy()
    ret_data = []
    for char in char_list:
        # data = data_.dropna(subset=[char])

        data[char_by] = data[char]
        data.loc[data[size_class] == 0, char_by] = None  # > NYSE20
        data[char_class] = classify(data, char, split, ascending=False, by=char_by)

        if weighting == 'ew':
            ret_data_ = one_dim_sort(data, char_class, target)
        elif weighting == 'vw':
            ret_data_ = one_dim_sort(data, char_class, target, weight_col=size_col)
        elif weighting == 'vw_cap':
            ret_data_ = one_dim_sort(data, char_class, target, weight_col='me_cap')

        ret_data_[['signal', 'n_stocks']] = data.groupby(['date', char_class])[char].agg(['mean', 'count'])

        ret_data_ = append_long_short(ret_data_)

        ret_data.append(ret_data_)
        elapsed_time(f'characteristic portfolio created: {char}')

    return pd.concat(ret_data, axis=1, keys=char_list, names=['characteristic', 'attributes'])


################################################################################
#
# Test functions to verify the code against JKP SAS code.
#
################################################################################
def read_jkp_data():
    data = pd.read_pickle('jkp/kelly_2000.pickle')
    data = data.set_index(['date', 'permno'])
    data['target'] = data.groupby('permno').ret_exc.shift(-1)
    data = data.rename(columns={'primary_sec': 'primary', 'crsp_exchcd': 'exchcd'})
    data['primary'] = data['primary'].astype(bool)
    data = data[(data.common == 1) & (data.ret_lag_dif == 1) & (data.obs_main == 1)]

    return data


def covert_jkp_factor_portfolios():
    fdata = pd.read_csv('jkp/AP_FACTORS_MONTHLY.csv')
    fdata = fdata[fdata.excntry == 'USA']
    fdata['date'] = pd.to_datetime(fdata.eom.astype('str'))
    fdata.set_index('date', inplace=True)
    fdata.to_pickle('jkp/AP_FACTORS_MONTHLY.pickle')

    fdata = pd.read_csv('jkp/AP_FACTORS_DAILY.csv')
    fdata = fdata[fdata.excntry == 'USA']
    fdata['date'] = pd.to_datetime(fdata.date.astype('str'))
    fdata.set_index('date', inplace=True)
    fdata.to_pickle('jkp/AP_FACTORS_DAILY.pickle')


def test_factor_portfolios():
    data = pd.read_pickle('pyanomaly_new1.pickle')
    # data = read_jkp_data()  # use data generated by jkp

    factors, hml, inv, roe = make_factor_portfolios(data)

    factors2 = pd.read_csv('jkp/AP_FACTORS_MONTHLY.csv')
    factors2['date'] = pd.to_datetime(factors2.eom.astype('str'))
    factors2['date'] = factors2['date'].shift(1)
    factors2.set_index('date', inplace=True)
    df = pd.merge(factors, factors2, on='date')

    compare_data(df)

    return df, factors, hml, inv, roe


def test_char_portfolio():
    # data = pd.read_pickle('pyanomaly_new1.pickle')
    data = read_jkp_data()  # use data generated by jkp

    char_list = ['ret_12_1', 'ret_1_0']
    fps, hmls = {}, {}
    for weighting in ['ew', 'vw', 'vw_cap']:
        fp = make_char_portfolios(data, char_list, weighting)
        hml = fp[fp.index.get_level_values(1) == 3]  # 0: h, 1:m, 2:l, 3: hml
        hml = hml.reset_index(level=-1, drop=True)
        hml = hml.stack(level=0).sort_index(level=[-1, 0])
        fps[weighting] = fp
        hmls[weighting] = hml

    hml = hmls['ew'].rename(columns={'target': 'ret_ew'})
    hml['ret_vw'] = hmls['vw']['target']
    hml['ret_vw_cap'] = hmls['vw_cap']['target']

    # JKP result
    hml_kelly = pd.read_csv('jkp/hml_pfs.csv')
    hml_kelly['date'] = pd.to_datetime(hml_kelly.eom.astype(str))
    hml_kelly['date'] = hml_kelly.groupby('characteristic').date.shift(1)  # shift date to follow pyanomaly convention
    hml_kelly.set_index(['date', 'characteristic'], inplace=True)

    # Compare
    print('Comparison: ret_ew')
    ret_ew = pd.merge(hml['ret_ew'].unstack(level=1), hml_kelly['ret_ew'].unstack(level=1), on='date')
    compare_data(ret_ew)

    print('Comparison: ret_vw')
    ret_vw = pd.merge(hml['ret_vw'].unstack(level=1), hml_kelly['ret_vw'].unstack(level=1), on='date')
    compare_data(ret_vw)

    print('Comparison: ret_vw_cap')
    ret_vwc = pd.merge(hml['ret_vw_cap'].unstack(level=1), hml_kelly['ret_vw_cap'].unstack(level=1), on='date')
    compare_data(ret_vwc)

    return hml


if __name__ == '__main__':
    os.chdir('../')

    # df, factors, hml, inv, roe = test_factor_portfolios()

    # hml = test_char_portfolio()
    # sdate = None
    # factors_m, hml_m, inv_m, roe_m = make_factor_portfolios(monthly=True, sdate=sdate)
    # factors_d, hml_d, inv_d, roe_d = make_factor_portfolios(monthly=False, sdate=sdate)
    covert_jkp_factor_portfolios()
