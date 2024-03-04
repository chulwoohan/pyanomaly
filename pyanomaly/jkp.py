"""This module defines functions to generate firm characteristics, factors, and characteristic portfolios
replicating JKP's SAS code.
"""

from pyanomaly.globals import *
# config.REPLICATE_JKP = True

from pyanomaly.analytics import *
from pyanomaly.fileio import write_to_file, read_from_file
from pyanomaly.characteristics import CRSPM, CRSPDRaw, FUNDQ, FUNDA, Merge

################################################################################
#
# Firm characteristics generation
#
################################################################################

def generate_firm_characterisitcs(fname=None):
    """Firm characteristics generation (Replication of JKP's SAS code).

    This function generate the firm characteristics that appear in JKP (2021).
    We replicate JKP's SAS code as closely as possible.
    The output is saved to 'config.output_dir/fname.pickle`.

    It is assumed that

        - the raw data has been downloaded from WRDS. If not, call ``WRDS.download_all()``;
        - factor portfolios has been created. If not, call ``make_factor_portfolios()``.

    Args:
        fname: Output file name. If None, fname = 'merge'.
    """

    set_log_path('./log/generate_firmcharacterisitcs.log')
    elapsed_time()

    # Generate characteristics in 'jkp' column in mapping.xlsx.
    alias = 'jkp'
    # Start date.
    sdate = '1950-01-01'

    ###################################
    # CRSPM
    ###################################
    drawline()
    log('PROCESSING CRSPM')
    drawline()

    # Load raw data.
    crspm = CRSPM(alias=alias)
    crspm.load_data(sdate)

    # Preprocess data.
    crspm.filter_data()
    crspm.populate(freq=MONTHLY, method=None)
    crspm.update_variables()
    crspm.merge_with_factors()

    # Generate characteristics.
    crspm_chars = crspm.get_available_chars()
    crspm.create_chars(crspm_chars)

    # Save the output.
    crspm.postprocess()
    # crspm.save()

    ###################################
    # CRSPD
    ###################################
    drawline()
    log('PROCESSING CRSPD')
    drawline()

    # Load raw data.
    crspd = CRSPD(alias=alias)
    crspd.load_data(sdate)

    # Preprocess data.
    crspd.filter_data()
    crspd.update_variables()
    crspd.merge_with_factors()

    # Generate characteristics.
    crspd_chars = crspd.get_available_chars()
    crspd.create_chars(crspd_chars)

    # Save the output.
    crspd.postprocess()
    # crspd.save()

    ###################################
    # FUNDQ
    ###################################
    drawline()
    log('PROCESSING FUNDQ')
    drawline()

    # Load raw data.
    fundq = FUNDQ(alias=alias)
    fundq.load_data(sdate)

    # Preprocess data.
    fundq.remove_duplicates()
    fundq.convert_currency()
    fundq.populate(MONTHLY, limit=3, lag=4, new_date_col='date')
    fundq.create_qitems_from_yitems()
    fundq.update_variables()

    # Generate characteristics.
    fundq_chars = fundq.get_available_chars()
    fundq.create_chars(fundq_chars)

    # Save the output.
    fundq.postprocess()
    # fundq.save()

    ###################################
    # FUNDA
    ###################################
    drawline()
    log('PROCESSING FUNDA')
    drawline()

    # Load raw data.
    funda = FUNDA(alias=alias)
    funda.load_data(sdate)

    # Preprocess data.
    funda.convert_currency()
    funda.populate(MONTHLY, limit=12, lag=4, new_date_col='date')
    funda.merge_with_fundq(fundq)
    funda.update_variables()
    funda.add_crsp_me(crspm)

    # Generate characteristics.
    funda_chars = funda.get_available_chars()
    funda.create_chars(funda_chars)

    # Save the output.
    funda.postprocess()
    # funda.save()

    ###################################
    # Merge
    ###################################
    drawline()
    log('PROCESSING MERGE')
    drawline()

    # Merge all data together.
    merge = Merge()
    merge.preprocess(crspm, crspd, funda, fundq)

    # Generate characteristics.
    merge_chars = merge.get_available_chars()
    merge.create_chars(merge_chars)

    # Save the output.
    merge.postprocess()
    merge.save(fname)


################################################################################
#
# Factor portfolio generation
#
################################################################################

def prepare_data_for_factors(monthly=True, daily=True, sdate=None):
    """Create characteristics needed to make factors.

    Args:
        monthly: If True (False), generate data monthly (daily).
        sdate: Start date ('yyyy-mm-dd').
    """

    crspm = CRSPM()
    crspm.load_data(sdate)

    crspm.filter_data()
    crspm.populate(method=None)
    crspm.update_variables()
    keep_columns(crspm.data, ['gvkey', 'rf', 'exchcd', 'primary', 'me', 'me_company', 'exret'])

    fundq = FUNDQ()
    fundq.load_data(sdate)

    fundq.remove_duplicates()
    fundq.convert_currency()
    fundq.populate(MONTHLY, limit=3, lag=4, new_date_col='date')
    fundq.create_qitems_from_yitems()
    fundq.update_variables()

    fundq.create_chars(['niq_be'])

    funda = FUNDA()
    funda.load_data(sdate)

    funda.convert_currency()
    funda.populate(MONTHLY, limit=12, lag=4, new_date_col='date')
    funda.merge_with_fundq(fundq)
    funda.update_variables()
    funda.add_crsp_me(crspm)

    funda.create_chars(['at_gr1', 'be_me'])

    keep_columns(funda.data,['at_gr1', 'be_me'])
    keep_columns(fundq.data, ['niq_be'])

    if monthly:
        mdata = crspm.data
        mdata.reset_index(inplace=True)
        mdata.set_index(['date', 'gvkey'], inplace=True)

        mdata = merge(mdata, funda.data, on=['date', 'gvkey'], how='left')
        mdata = merge(mdata, fundq.data, on=['date', 'gvkey'], how='left')

        mdata.reset_index(inplace=True)
        mdata.set_index(['date', 'permno'], inplace=True)
    else:
        mdata = None

    if daily:
        crspd = CRSPDRaw()
        crspd.load_data(sdate)

        crspd.filter_data()
        crspd.update_variables()
        keep_columns(crspd.data, ['gvkey', 'rf', 'exchcd', 'primary', 'me', 'me_company', 'exret', 'ym'])

        ddata = crspd.data
        ddata.reset_index(inplace=True)
        ddata.set_index(['ym', 'gvkey'], inplace=True)

        ddata = merge(ddata, funda.data, on=['ym', 'gvkey'], right_on=['date', 'gvkey'], how='left')
        ddata = merge(ddata, fundq.data, on=['ym', 'gvkey'], right_on=['date', 'gvkey'], how='left')

        ddata.reset_index(inplace=True)
        ddata.set_index(['date', 'permno'], inplace=True)
    else:
        ddata = None

    return mdata, ddata


def make_factor_portfolio(data, char, ret_col, size_class, weight_col=None):
    """Make a factor portfolio.

    Args:
        data: DataFrame with index = date/id. Its columns should include `char`, `ret_col`, `size_class`, and
            `weight_col` (optional).
        char: Characteristic column to make a factor portfolio from.
        ret_col: Return column.
        size_class: Size class column.
        weight_col: Weight column. If None, stocks are equally weighted.

    Returns:
        DataFrame of (size x `char`) factor portfolios. Index = 'date',
        columns: ['bh', 'bl', 'bm', 'sh', 'sl', 'sm', 'hml', 'smb'].
    """

    char_split = [0.3, 0.7, 1.0]
    char_by = char + '_by'  # Column to determine quantile breakpoints: same as char but keeps only NYSE stocks.
    char_class = char + '_class'  # Class (group) column.

    # Classify stocks on `char`.
    data[char_by] = data[char].where(data.exchcd == 1)  # Keep only NYSE stocks.
    data[char_class] = classify(data, char_split, ascending=False, on=char, by=char_by)

    # 2-D sort. factors index = date/char_class/size_class, columns = [ret].
    factors = two_dim_sort(data, char_class, size_class, ret_col, weight_col=weight_col, output_dim=2)

    # Name portfolios.
    relabel_class(factors,  ['h', 'm', 'l', 'hml'], axis=0)
    relabel_class(factors, ['s', 'b', 'smb'], axis=1)
    factors = factors.unstack()
    factors.columns = factors.columns.map(''.join)

    factors['hml'] = (factors['shml'] + factors['bhml']) / 2
    factors['smb'] = (factors['smbh'] + factors['smbm'] + factors['smbl']) / 3
    drop_columns(factors, ['shml', 'bhml', 'smbh', 'smbm', 'smbl', 'smbhml'])

    elapsed_time(f'factor portfolio created: {char}')

    return factors


def make_factor_portfolios_(data):
    """Make factor portfolios.

    FF 3 and HXZ 4 factors are generated. The result is saved to `config.factors_monthly(daily)_fname`.

    Args:
        monthly: If True (False), generate data monthly (daily).
        sdate: Start date ('yyyy-mm-dd').

    Returns:
        Factor portfolio dataframe with index = 'date' and columns = ['mktrf', 'rf', 'smb_ff', 'hml', 'inv', 'roe',
        'smb_hxz'].
    """

    ret_col = 'futret'  # t+1 excess return
    size_col = 'me'  # We use security level me following JKP but think we should use me_company.

    # One-period ahead return.
    data[ret_col] = future_return(data.exret)
    # The futret at t is the return from t to t+1. Shift one period so that the return is from t-1 to t and
    # other variables are as of t.
    data = data.groupby(level=-1).shift(1)  # Shift data forward.

    ########################################
    # Filtering
    ########################################
    data.dropna(subset=[ret_col, size_col], inplace=True)  # non-missing return and me
    data = data[data.primary]  # primary securities only

    ########################################
    # Size split
    ########################################
    size_split = [0.5, 1.0]
    size_class = 'size_class'

    data['nyse_me'] = data[size_col].where(data.exchcd == 1)  # Keep only NYSE stocks.
    data[size_class] = classify(data, size_split, ascending=True, on=size_col, by='nyse_me')

    ########################################
    # Factor portfolios
    ########################################
    # Market pf.
    factors = weighted_mean(data, ret_col, size_col, 'date')
    factors.rename(columns={ret_col: 'mktrf'}, inplace=True)
    factors['rf'] = data.groupby('date').rf.first()

    # FF factors.
    hml = make_factor_portfolio(data, 'be_me', ret_col, size_class, size_col)
    factors[['smb_ff', 'hml']] = hml[['smb', 'hml']]

    # HXZ factors.
    inv = make_factor_portfolio(data, 'at_gr1', ret_col, size_class, size_col)
    roe = make_factor_portfolio(data, 'niq_be', ret_col, size_class, size_col)
    factors['inv'] = -inv['hml']
    factors['roe'] = roe['hml']
    factors['smb_hxz'] = (inv['smb'] + roe['smb']) / 2

    return factors


def make_factor_portfolios(monthly=True, daily=True, sdate=None):
    """Make factor portfolios.

    FF 3 and HXZ 4 factors are generated. The result is saved to `config.factors_monthly(daily)_fname`.

    Args:
        monthly: If True (False), generate data monthly (daily).
        sdate: Start date ('yyyy-mm-dd').

    Returns:
        Factor portfolio dataframe with index = 'date' and columns = ['mktrf', 'rf', 'smb_ff', 'hml', 'inv', 'roe',
        'smb_hxz'].
    """
    mdata, ddata = prepare_data_for_factors(monthly, daily, sdate)
    # mdata = pd.read_pickle('mdata.pickle')
    # ddata = pd.read_pickle('ddata.pickle')

    # data = read_jkp_data()
    if monthly:
        elapsed_time('making factor portfolios monthly...')
        mfactors = make_factor_portfolios_(mdata)
        write_to_file(mfactors, config.factors_monthly_fname)

    if daily:
        elapsed_time('making factor portfolios daily...')
        dfactors = make_factor_portfolios_(ddata)
        write_to_file(dfactors, config.factors_daily_fname)

    if monthly and daily:
        return mfactors, dfactors
    elif monthly:
        return mfactors
    elif daily:
        return dfactors

################################################################################
#
# Characteristic portfolios
#
################################################################################
def make_char_portfolios(data, char_list, weighting):
    """Make characteristic portfolios.

    Args:
        data: DataFrame of firm characteristics.
        char_list: List of characteristics to generate.
        weighting: 'ew' (Equal-weight), 'vw' (Value-weight), or 'vw_cap' (Value-weight capped at 0.8 NYSE-size quantile).

    Returns:
        Characteristic portfolio DataFrame with index = date/class and columns = [char, 'ret', 'signal', 'n_firms'].
        The class values are one of 'h', 'm', 'l', 'hml'.
    """

    elapsed_time(f'make_char_portfolios. weighting: {weighting}')

    ret_col = 'futret'  # t+1 excess return
    size_col = 'me'  # We use security level me following JKP but think we should use me_company.

    # One-period ahead return.
    data[ret_col] = future_return(data.exret)
    # The futret at t is the return from t to t+1. Shift one period so that the return is from t-1 to t and
    # other variables are as of t.
    data = data.groupby(level=-1).shift(1)  # Shift data forward.

    ########################################
    # Filtering
    ########################################
    data = data.dropna(subset=[ret_col, size_col])  # non-missing t+1 return and me
    data = data[data.primary == True]  # primary securities only

    ########################################
    # Size split
    ########################################
    data['nyse_me'] = np.where(data.exchcd == 1, data[size_col], np.nan)  # NYSE me

    if weighting == 'vw':
        weight_col = size_col
    elif weighting == 'vw_cap':
        data['me_cap'] = winsorize(data[size_col], (None, 0.2), by_array=data['nyse_me'])
        weight_col = 'me_cap'
    else: # weighting == 'ew':
        weight_col = None

    elapsed_time('size_class generated.')

    ########################################
    # Characteristic portfolios
    ########################################
    # Add identifier for the bottom 20%.
    size_split = [0.2, 1.0]
    size_class = 'size_class'
    data[size_class] = classify(data, size_split, ascending=True, on=size_col, by='nyse_me')

    split = 3
    char_by = 'char_by'
    group_col = 'group'
    labels = ['h', 'm', 'l', 'hml']

    char_portfolios = []
    for char in char_list:
        data[char_by] = data[char].where(data[size_class] > 0)  # NYSE-characteristic

        # Classify data on char.
        data[group_col] = classify(data, split, ascending=False, on=char, by=char_by)

        # Make quantile portfolios.
        portfolios = one_dim_sort(data, group_col, ret_col, weight_col=weight_col)
        portfolios[['signal', 'n_firms']] = one_dim_sort(data, group_col, char, function=['mean', 'count'])

        relabel_class(portfolios, labels)
        portfolios = portfolios.rename(columns={ret_col: 'ret'})
        portfolios['characteristic'] = char
        char_portfolios.append(portfolios.reset_index(group_col))

        elapsed_time(f'Characteristic portfolio created: {char}')

    return pd.concat(char_portfolios)


def make_char_portfolios_all():
    alias = 'jkp'

    data = read_from_file('merge_full')

    # Choose characteristics.
    char_map = pd.read_excel(config.mapping_file_path)
    char_list = list(char_map['jkp'].dropna())
    char_list = [char for char in char_list if char in data]  # Only those in data.

    # Equal-weight
    char_portfolios = make_char_portfolios(data, char_list, 'ew')
    write_to_file(char_portfolios, 'char_portfolios_ew')

    # Value-weight
    char_portfolios = make_char_portfolios(data, char_list, 'vw')
    write_to_file(char_portfolios, 'char_portfolios_vw')

    # Capped value-weight
    char_portfolios = make_char_portfolios(data, char_list, 'vw_cap')
    write_to_file(char_portfolios, 'char_portfolios_vw_cap')


################################################################################
#
# Aux. function for testing. (IGNORE)
#
# The fuctions below may not work on your side as they read some files I created
# for testing purposes.
################################################################################

def read_jkp_data():
    # Read JKP data.

    data = pd.read_pickle('jkp/kelly_2000.pickle')
    # data = pd.read_pickle('jkp/world_data.pickle')
    # data.date = to_month_end(data.date)
    data = data.set_index(['date', 'permno'])
    data['exret'] = data.ret_exc
    # data['futret'] = data.groupby('permno').ret_exc.shift(-1)
    data = data.rename(columns={'primary_sec': 'primary', 'crsp_exchcd': 'exchcd'})
    data['primary'] = data['primary'].astype(bool)
    data = data[(data.common == 1) & (data.ret_lag_dif == 1) & (data.obs_main == 1)]

    return data


def covert_jkp_factor_portfolios():
    # Convert factor files generated by JKP's SAS code to pickle files.

    fdata = pd.read_csv('AP_FACTORS_MONTHLY.csv')
    fdata = fdata[fdata.excntry == 'USA']
    fdata['date'] = pd.to_datetime(fdata.eom.astype('str'))
    fdata.set_index('date', inplace=True)
    fdata.to_pickle('output/factors_monthly_jkp.pickle')

    fdata = pd.read_csv('AP_FACTORS_DAILY.csv')
    fdata = fdata[fdata.excntry == 'USA']
    fdata['date'] = pd.to_datetime(fdata.date.astype('str'))
    fdata.set_index('date', inplace=True)
    fdata.to_pickle('output/factors_daily_jkp.pickle')


def test_factor_portfolios(monthly=True):
    # Test factor portfolio generation.

    factors = make_factor_portfolios(monthly=monthly)

    if monthly:
        factors_jkp = read_from_file('factors_monthly_jkp')
    else:
        factors_jkp = read_from_file('factors_daily_jkp')

    compare_data(factors, factors_jkp, tolerance=0.1)
    return factors


def test_char_portfolio():
    # Test characteristic portfolio generation.

    char_list = ['ret_1_0', 'ret_12_1']

    data = read_jkp_data()  # use data generated by jkp

    # JKP result
    hml_jkp = pd.read_csv('jkp/hml_pfs.csv')
    hml_jkp['date'] = pd.to_datetime(hml_jkp.eom.astype(str))
    hml_jkp = hml_jkp.set_index('date')
    hml_jkp = hml_jkp[hml_jkp.characteristic.isin(char_list)]

    hml = None
    for weighting in ['vw', 'ew', 'vw_cap']:
        char_pfs = make_char_portfolios(data, char_list, weighting)
        if hml is None:
            hml = char_pfs[char_pfs['group'] == 'hml']
            hml = hml.rename(columns={'ret': 'ret_' + weighting})
        else:
            hml['ret_' + weighting] = char_pfs.loc[char_pfs['group'] == 'hml', 'ret']

    for char in char_list:
        print('\n', char)
        compare_data(hml[hml.characteristic==char], hml_jkp[hml_jkp.characteristic == char], tolerance=0.1)


if __name__ == '__main__':
    os.chdir('../')

    # make_factor_portfolios(monthly=True)
    # make_factor_portfolios(monthly=False)
    # generate_firm_characterisitcs()
    make_char_portfolios_all()

    # test_factor_portfolios(monthly=False)
    # test_char_portfolio()

