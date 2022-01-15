"""Fama-French Factors.

This module defines a function to generate Fama-French factors.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyanomaly.globals import *
from pyanomaly.datatools import *
from pyanomaly.analytics import classify, two_dim_sort, relabel_class
from pyanomaly.characteristics import FUNDA, CRSPM
from pyanomaly.wrdsdata import WRDS


def make_ff_factors():
    """Generate Fama-French 3 factors.

    This function refers to the WRDS code, but the results are slightly different as the code is written under
    the architecture of PyAnomaly. Compared to the data from the K. French website, HML has a correlation of 0.967,
    and SMB has a correlation of 0.989. Compared to the WRDS code, HML has a correlation of 0.991,
    and SMB has a correlation of 0.993.

    WRDS code:
    https://wrds-www.wharton.upenn.edu/pages/support/applications/python-replications/fama-french-factors-python/

    Major differences from WRDS code:

    1. Primary stock identification: for our method, refer to ``wrds.add_gvkey_to_crsp()``.
    2. Delist return: for our method, refer to ``wrds.merge_sf_with_seall()``.

    .. todo::
        Add mktrf, rf, and the rest of the 5 factors.

    Returns:
        factors, number of firms.

        * factors (DataFrame): index = 'date', columns: 'bh', 'bm', 'bl', 'sh', 'sm', 'sl', 'hml', 'smb'.
        * number of firms (DataFrame): index = 'date', columns: 'bh', 'bm', 'bl', 'sh', 'sm', 'sl', 'hml', 'smb'.
    """

    # CRSPM
    crspm = CRSPM()
    crspm.load_data()

    crspm.filter(('exchcd', 'in', [1, 2, 3]))
    crspm.filter(('shrcd', 'in', [10, 11]))
    crspm.update_variables()

    cm = crspm.data.reset_index()
    cm['me'] = cm.me_company  # Use company level me.
    cm['me_1'] = cm.groupby('permno').me.shift(1)  # 1-month lagged me.
    # For me_1, the following code is more rigorous as there can be missing months. But the difference seems negligible.
    # cm0 = cm[['date', 'permno', 'me']].rename(columns={'me': 'me_1'})
    # cm0['date'] = add_months(cm0.date, 1)
    # cm = cm.merge(cm0, on=['date', 'permno'])

    # Primary security only
    cm = cm[(cm.primary == True) & (cm.me > 0)]

    # FUNDA
    funda = FUNDA()
    funda.load_data()
    funda.update_variables()
    fa = funda.data.reset_index()
    fa['year'] = fa.datadate.dt.year
    fa = fa.drop_duplicates(['gvkey', 'year'], keep='last')  # Fiscal year change can cause dup. Keep the most recent.
    fa['rcount'] = fa.groupby('gvkey').cumcount()  # reset rcount (since some rows are dropped)
    fa = fa[['gvkey', 'datadate', 'be', 'rcount', 'year']]

    # Filter funda
    fa = fa[(fa.rcount >= 1) & (fa.be > 0)]

    # ME for BEME and SIZE
    # BEME: t-1 year-end me, SIZE: t year June me.
    me_jun = cm.loc[cm.date.dt.month == 6, ['date', 'gvkey', 'permno', 'me', 'exchcd']].rename(columns={'me': 'me_jun'})
    me_jun['year'] = me_jun.date.dt.year - 1
    fa = fa.merge(me_jun, on=['year', 'gvkey'])

    me_dec = cm.loc[cm.date.dt.month == 12, ['date', 'gvkey', 'permno', 'me']].rename(columns={'me': 'me_dec'})
    me_dec['year'] = me_dec.date.dt.year
    fa = fa.merge(me_dec, on=['year', 'gvkey', 'permno'])

    # me_jun = cm.loc[cm.date.dt.month == 6, ['date', 'permno', 'gvkey', 'me', 'exchcd']].rename(columns={'me': 'me_jun'})
    # me_jun['year'] = me_jun.date.dt.year - 1
    # me_dec = cm.loc[cm.date.dt.month == 12, ['date', 'permno', 'me']].rename(columns={'me': 'me_dec'})
    # me_dec['year'] = me_dec.date.dt.year
    # me = pd.merge(me_dec, me_jun, on=['year', 'permno'])
    # fa = fa.merge(me, on=['year', 'gvkey'])

    fa['be_me'] = fa.be / fa.me_dec
    fa = fa.set_index(['year', 'permno'])

    # SIZE
    size_col = 'me_jun'
    size_split = [0.5, 1.0]
    size_labels = ['s', 'b', 'smb']
    size_by = size_col + '_by'
    fa[size_by] = np.where((fa.exchcd == 1) & (fa.rcount > 1), fa[size_col], np.nan)  # NYSE and at least 2 years in funda
    fa['size_class'] = classify(fa[size_col], size_split, ascending=True, by_array=fa[size_by])

    # BM
    bm_col = 'be_me'
    bm_split = [0.3, 0.7, 1.0]
    bm_labels = ['h', 'm', 'l', 'hml']
    bm_by = bm_col + '_by'
    fa[bm_by] = np.where((fa.exchcd == 1) & (fa.rcount > 1), fa[bm_col], np.nan)  # NYSE and at least 2 years in funda
    fa['bm_class'] = classify(fa[bm_col], bm_split, ascending=False, by_array=fa[bm_by])

    # Remove unclassified rows. This happens if there are no NYSE stocks with at least 2 years in funda.
    fa = fa[fa['size_class'].isin([0, 1]) & fa['bm_class'].isin([0, 1, 2])]

    # Merge annual fa with monthly cm
    cm['year'] = add_months(cm.date, -18).dt.year
    cm = pd.merge(cm, fa[fa.columns.difference(cm.columns)], on=['permno', 'year'])
    cm = cm.set_index(['date', 'permno'])

    # Double sort on size and bm.
    ff_factors = two_dim_sort(cm, 'bm_class', 'size_class', 'ret', weight_col='me_1', output_dim=2)

    relabel_class(ff_factors, bm_labels, axis=0)
    relabel_class(ff_factors, size_labels, axis=1)
    ff_factors = ff_factors.unstack()
    ff_factors.columns = ff_factors.columns.map(''.join)

    # smb and hml factors
    ff_factors['hml'] = (ff_factors['shml'] + ff_factors['bhml']) / 2
    ff_factors['smb'] = (ff_factors['smbh'] + ff_factors['smbm'] + ff_factors['smbl']) / 3
    drop_columns(ff_factors, ['shml', 'bhml', 'smbh', 'smbm', 'smbl', 'smbhml'])

    # Firm count
    ff_nfirms = two_dim_sort(cm, 'bm_class', 'size_class', 'ret', function='count', add_long_short=False, output_dim=2)

    relabel_class(ff_nfirms, bm_labels, axis=0)
    relabel_class(ff_nfirms, size_labels, axis=1)
    ff_nfirms = ff_nfirms.unstack()
    ff_nfirms.columns = ff_nfirms.columns.map(''.join)

    ff_nfirms['hml'] = ff_nfirms['bh'] + ff_nfirms['sh'] + ff_nfirms['bl'] + ff_nfirms['sl']
    ff_nfirms['smb'] = ff_nfirms['sh'] + ff_nfirms['sm'] + ff_nfirms['sl'] + \
                       ff_nfirms['bh'] + ff_nfirms['bm'] + ff_nfirms['bl']

    return ff_factors, ff_nfirms



###############################################################################
#
# WRDS Code
#
###############################################################################
import datetime as dt
import wrds
from scipy import stats

def make_ff_factors_wrds():
    """
    This is a simple copy of the WRDS code.

    https://wrds-www.wharton.upenn.edu/pages/support/applications/python-replications/fama-french-factors-python/

    Returns:
        factors, number of firms.
    """
    ###################
    # Connect to WRDS #
    ###################
    conn = wrds.Connection()

    ###################
    # Compustat Block #
    ###################
    comp = conn.raw_sql("""
                        select gvkey, datadate, at, pstkl, txditc,
                        pstkrv, seq, pstk
                        from comp.funda
                        where indfmt='INDL' 
                        and datafmt='STD'
                        and popsrc='D'
                        and consol='C'
                        and datadate >= '01/01/1959'
                        """, date_cols=['datadate'])

    comp['year'] = comp['datadate'].dt.year

    # create preferrerd stock
    comp['ps'] = np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
    comp['ps'] = np.where(comp['ps'].isnull(), comp['pstk'], comp['ps'])
    comp['ps'] = np.where(comp['ps'].isnull(), 0, comp['ps'])
    comp['txditc'] = comp['txditc'].fillna(0)

    # create book equity
    comp['be']=comp['seq']+comp['txditc']-comp['ps']
    comp['be']=np.where(comp['be']>0, comp['be'], np.nan)

    # number of years in Compustat
    comp = comp.sort_values(by=['gvkey', 'datadate'])
    comp['count'] = comp.groupby(['gvkey']).cumcount()

    comp = comp[['gvkey', 'datadate', 'year', 'be', 'count']]

    ###################
    # CRSP Block      #
    ###################
    # sql similar to crspmerge macro
    crsp_m = conn.raw_sql("""
                          select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
                          a.ret, a.retx, a.shrout, a.prc
                          from crsp.msf as a
                          left join crsp.msenames as b
                          on a.permno=b.permno
                          and b.namedt<=a.date
                          and a.date<=b.nameendt
                          where a.date between '01/01/1959' and '12/31/2017'
                          and b.exchcd between 1 and 3
                          """, date_cols=['date'])

    # change variable format to int
    crsp_m[['permco', 'permno', 'shrcd', 'exchcd']] = crsp_m[['permco', 'permno', 'shrcd', 'exchcd']].astype(int)

    # Line up date to be end of month
    crsp_m['jdate'] = crsp_m['date'] + MonthEnd(0)

    # add delisting return
    dlret = conn.raw_sql("""
                         select permno, dlret, dlstdt 
                         from crsp.msedelist
                         """, date_cols=['dlstdt'])

    dlret.permno = dlret.permno.astype(int)
    # dlret['dlstdt']=pd.to_datetime(dlret['dlstdt'])
    dlret['jdate'] = dlret['dlstdt'] + MonthEnd(0)

    crsp = pd.merge(crsp_m, dlret, how='left', on=['permno', 'jdate'])
    crsp['dlret'] = crsp['dlret'].fillna(0)
    crsp['ret'] = crsp['ret'].fillna(0)

    # retadj factors in the delisting returns
    crsp['retadj'] = (1 + crsp['ret']) * (1 + crsp['dlret']) - 1

    # calculate market equity
    crsp['me'] = crsp['prc'].abs() * crsp['shrout']
    crsp = crsp.drop(['dlret', 'dlstdt', 'prc', 'shrout'], axis=1)
    crsp = crsp.sort_values(by=['jdate', 'permco', 'me'])

    ### Aggregate Market Cap ###
    # sum of me across different permno belonging to same permco a given date
    crsp_summe = crsp.groupby(['jdate', 'permco'])['me'].sum().reset_index()

    # largest mktcap within a permco/date
    crsp_maxme = crsp.groupby(['jdate', 'permco'])['me'].max().reset_index()

    # join by jdate/maxme to find the permno
    crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=['jdate', 'permco', 'me'])

    # drop me column and replace with the sum me
    crsp1 = crsp1.drop(['me'], axis=1)

    # join with sum of me to get the correct market cap info
    crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=['jdate', 'permco'])

    # sort by permno and date and also drop duplicates
    crsp2 = crsp2.sort_values(by=['permno', 'jdate']).drop_duplicates()

    # keep December market cap
    crsp2['year'] = crsp2['jdate'].dt.year
    crsp2['month'] = crsp2['jdate'].dt.month
    decme = crsp2[crsp2['month'] == 12]
    decme = decme[['permno', 'date', 'jdate', 'me', 'year']].rename(columns={'me': 'dec_me'})

    ### July to June dates
    crsp2['ffdate'] = crsp2['jdate'] + MonthEnd(-6)
    crsp2['ffyear'] = crsp2['ffdate'].dt.year
    crsp2['ffmonth'] = crsp2['ffdate'].dt.month
    crsp2['1+retx'] = 1 + crsp2['retx']
    crsp2 = crsp2.sort_values(by=['permno', 'date'])

    # cumret by stock
    crsp2['cumretx'] = crsp2.groupby(['permno', 'ffyear'])['1+retx'].cumprod()

    # lag cumret
    crsp2['lcumretx'] = crsp2.groupby(['permno'])['cumretx'].shift(1)

    # lag market cap
    crsp2['lme'] = crsp2.groupby(['permno'])['me'].shift(1)

    # if first permno then use me/(1+retx) to replace the missing value
    crsp2['count'] = crsp2.groupby(['permno']).cumcount()
    crsp2['lme'] = np.where(crsp2['count'] == 0, crsp2['me'] / crsp2['1+retx'], crsp2['lme'])

    # baseline me
    mebase = crsp2[crsp2['ffmonth'] == 1][['permno', 'ffyear', 'lme']].rename(columns={'lme': 'mebase'})

    # merge result back together
    crsp3 = pd.merge(crsp2, mebase, how='left', on=['permno', 'ffyear'])
    crsp3['wt'] = np.where(crsp3['ffmonth'] == 1, crsp3['lme'], crsp3['mebase'] * crsp3['lcumretx'])

    decme['year'] = decme['year'] + 1
    decme = decme[['permno', 'year', 'dec_me']]

    # Info as of June
    crsp3_jun = crsp3[crsp3['month'] == 6]

    crsp_jun = pd.merge(crsp3_jun, decme, how='inner', on=['permno', 'year'])
    crsp_jun = crsp_jun[
        ['permno', 'date', 'jdate', 'shrcd', 'exchcd', 'retadj', 'me', 'wt', 'cumretx', 'mebase', 'lme', 'dec_me']]
    crsp_jun = crsp_jun.sort_values(by=['permno', 'jdate']).drop_duplicates()

    #######################
    # CCM Block           #
    #######################
    ccm = conn.raw_sql("""
                      select gvkey, lpermno as permno, linktype, linkprim, 
                      linkdt, linkenddt
                      from crsp.ccmxpf_linktable
                      where substr(linktype,1,1)='L'
                      and (linkprim ='C' or linkprim='P')
                      """, date_cols=['linkdt', 'linkenddt'])

    # if linkenddt is missing then set to today date
    ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

    ccm1 = pd.merge(comp[['gvkey', 'datadate', 'be', 'count']], ccm, how='left', on=['gvkey'])
    ccm1['yearend'] = ccm1['datadate'] + pd.tseries.offsets.YearEnd(0)
    ccm1['jdate'] = ccm1['yearend'] + pd.tseries.offsets.MonthEnd(6)

    # set link date bounds
    ccm2 = ccm1[(ccm1['jdate'] >= ccm1['linkdt']) & (ccm1['jdate'] <= ccm1['linkenddt'])]
    ccm2 = ccm2[['gvkey', 'permno', 'datadate', 'yearend', 'jdate', 'be', 'count']]

    # link comp and crsp
    ccm_jun = pd.merge(crsp_jun, ccm2, how='inner', on=['permno', 'jdate'])
    ccm_jun['beme'] = ccm_jun['be'] * 1000 / ccm_jun['dec_me']

    # select NYSE stocks for bucket breakdown
    # exchcd = 1 and positive beme and positive me and shrcd in (10,11) and at least 2 years in comp
    nyse = ccm_jun[(ccm_jun['exchcd'] == 1) & (ccm_jun['beme'] > 0) & (ccm_jun['me'] > 0) & \
                   (ccm_jun['count'] >= 1) & ((ccm_jun['shrcd'] == 10) | (ccm_jun['shrcd'] == 11))]

    # size breakdown
    nyse_sz = nyse.groupby(['jdate'])['me'].median().to_frame().reset_index().rename(columns={'me': 'sizemedn'})

    # beme breakdown
    nyse_bm = nyse.groupby(['jdate'])['beme'].describe(percentiles=[0.3, 0.7]).reset_index()
    nyse_bm = nyse_bm[['jdate', '30%', '70%']].rename(columns={'30%': 'bm30', '70%': 'bm70'})

    nyse_breaks = pd.merge(nyse_sz, nyse_bm, how='inner', on=['jdate'])

    # join back size and beme breakdown
    ccm1_jun = pd.merge(ccm_jun, nyse_breaks, how='left', on=['jdate'])

    # function to assign sz and bm bucket
    def sz_bucket(row):
        if row['me'] == np.nan:
            value = ''
        elif row['me'] <= row['sizemedn']:
            value = 'S'
        else:
            value = 'B'
        return value

    def bm_bucket(row):
        if 0 <= row['beme'] <= row['bm30']:
            value = 'L'
        elif row['beme'] <= row['bm70']:
            value = 'M'
        elif row['beme'] > row['bm70']:
            value = 'H'
        else:
            value = ''
        return value

    # assign size portfolio
    ccm1_jun['szport'] = np.where((ccm1_jun['beme'] > 0) & (ccm1_jun['me'] > 0) & (ccm1_jun['count'] >= 1),
                                  ccm1_jun.apply(sz_bucket, axis=1), '')

    # assign book-to-market portfolio
    ccm1_jun['bmport'] = np.where((ccm1_jun['beme'] > 0) & (ccm1_jun['me'] > 0) & (ccm1_jun['count'] >= 1),
                                  ccm1_jun.apply(bm_bucket, axis=1), '')

    # create positivebmeme and nonmissport variable
    ccm1_jun['posbm'] = np.where((ccm1_jun['beme'] > 0) & (ccm1_jun['me'] > 0) & (ccm1_jun['count'] >= 1), 1, 0)
    ccm1_jun['nonmissport'] = np.where((ccm1_jun['bmport'] != ''), 1, 0)

    # store portfolio assignment as of June
    june = ccm1_jun[['permno', 'date', 'jdate', 'bmport', 'szport', 'posbm', 'nonmissport']]
    june['ffyear'] = june['jdate'].dt.year

    # merge back with monthly records
    crsp3 = crsp3[['date', 'permno', 'shrcd', 'exchcd', 'retadj', 'me', 'wt', 'cumretx', 'ffyear', 'jdate']]
    ccm3 = pd.merge(crsp3,
                    june[['permno', 'ffyear', 'szport', 'bmport', 'posbm', 'nonmissport']], how='left',
                    on=['permno', 'ffyear'])

    # keeping only records that meet the criteria
    ccm4 = ccm3[(ccm3['wt'] > 0) & (ccm3['posbm'] == 1) & (ccm3['nonmissport'] == 1) &
                ((ccm3['shrcd'] == 10) | (ccm3['shrcd'] == 11))]

    ############################
    # Form Fama French Factors #
    ############################

    # function to calculate value weighted return
    def wavg(group, avg_name, weight_name):
        d = group[avg_name]
        w = group[weight_name]
        try:
            return (d * w).sum() / w.sum()
        except ZeroDivisionError:
            return np.nan

    # value-weigthed return
    vwret = ccm4.groupby(['jdate', 'szport', 'bmport']).apply(wavg, 'retadj', 'wt').to_frame().reset_index().rename(
        columns={0: 'vwret'})
    vwret['sbport'] = vwret['szport'] + vwret['bmport']

    # firm count
    vwret_n = ccm4.groupby(['jdate', 'szport', 'bmport'])['retadj'].count().reset_index().rename(
        columns={'retadj': 'n_firms'})
    vwret_n['sbport'] = vwret_n['szport'] + vwret_n['bmport']

    # tranpose
    ff_factors = vwret.pivot(index='jdate', columns='sbport', values='vwret').reset_index()
    ff_nfirms = vwret_n.pivot(index='jdate', columns='sbport', values='n_firms').reset_index()
    # create SMB and HML factors
    ff_factors['WH'] = (ff_factors['BH'] + ff_factors['SH']) / 2
    ff_factors['WL'] = (ff_factors['BL'] + ff_factors['SL']) / 2
    ff_factors['WHML'] = ff_factors['WH'] - ff_factors['WL']

    ff_factors['WB'] = (ff_factors['BL'] + ff_factors['BM'] + ff_factors['BH']) / 3
    ff_factors['WS'] = (ff_factors['SL'] + ff_factors['SM'] + ff_factors['SH']) / 3
    ff_factors['WSMB'] = ff_factors['WS'] - ff_factors['WB']
    ff_factors = ff_factors.rename(columns={'jdate': 'date'})

    # n firm count
    ff_nfirms['H'] = ff_nfirms['SH'] + ff_nfirms['BH']
    ff_nfirms['L'] = ff_nfirms['SL'] + ff_nfirms['BL']
    ff_nfirms['HML'] = ff_nfirms['H'] + ff_nfirms['L']

    ff_nfirms['B'] = ff_nfirms['BL'] + ff_nfirms['BM'] + ff_nfirms['BH']
    ff_nfirms['S'] = ff_nfirms['SL'] + ff_nfirms['SM'] + ff_nfirms['SH']
    ff_nfirms['SMB'] = ff_nfirms['B'] + ff_nfirms['S']
    ff_nfirms['TOTAL'] = ff_nfirms['SMB']
    ff_nfirms = ff_nfirms.rename(columns={'jdate': 'date'})

    ###################
    # Compare With FF #
    ###################
    _ff = conn.get_table(library='ff', table='factors_monthly')
    _ff = _ff[['date', 'smb', 'hml']]
    _ff['date'] = _ff['date'] + MonthEnd(0)

    _ffcomp = pd.merge(_ff, ff_factors[['date', 'WSMB', 'WHML']], how='inner', on=['date'])
    _ffcomp70 = _ffcomp[_ffcomp['date'] >= '01/01/1970']
    print(stats.pearsonr(_ffcomp70['smb'], _ffcomp70['WSMB']))
    print(stats.pearsonr(_ffcomp70['hml'], _ffcomp70['WHML']))

    plt.figure(figsize=(16, 12))
    plt.suptitle('Comparison of Results', fontsize=20)

    ax1 = plt.subplot(211)
    ax1.set_title('SMB', fontsize=15)
    ax1.set_xlim([dt.datetime(1962, 6, 1), dt.datetime(2017, 12, 31)])
    ax1.plot(_ffcomp['smb'], 'r--', _ffcomp['WSMB'], 'b-')
    ax1.legend(('smb', 'WSMB'), loc='upper right', shadow=True)

    ax2 = plt.subplot(212)
    ax2.set_title('HML', fontsize=15)
    ax2.plot(_ffcomp['hml'], 'r--', _ffcomp['WHML'], 'b-')
    ax2.set_xlim([dt.datetime(1962, 6, 1), dt.datetime(2017, 12, 31)])
    ax2.legend(('hml', 'WHML'), loc='upper right', shadow=True)

    plt.subplots_adjust(top=0.92, hspace=0.2)

    plt.show()

    ff_factors.set_index('date', inplace=True)
    ff_factors = ff_factors.rename(
        columns={'BH': 'bh', 'BL': 'bl', 'BM': 'bm', 'SH': 'sh', 'SL': 'sl', 'SM': 'sm', 'WH': 'h', 'WL': 'l',
                 'WHML': 'hml', 'WB': 'b', 'WS': 's', 'WSMB': 'smb'})

    ff_nfirms.set_index('date', inplace=True)
    ff_nfirms = ff_nfirms.rename(
        columns={'BH': 'bh', 'BL': 'bl', 'BM': 'bm', 'SH': 'sh', 'SL': 'sl', 'SM': 'sm', 'H': 'h', 'L': 'l',
                 'HML': 'hml', 'B': 'b', 'S': 's', 'SMB': 'smb'})


    return ff_factors, ff_nfirms


if __name__ == '__main__':
    os.chdir('../')

    ##############################
    # Generate factors
    ##############################
    # ff factors from pyanomaly
    ff_factors1, ff_nfirms1 = make_ff_factors()

    # ff factors from WRDS code
    ff_factors2, ff_nfirms2 = make_ff_factors_wrds()

    # ff factors from French website
    ff_factors3 = WRDS.read_data('factors_monthly')
    ff_factors3['date'] = to_month_end(ff_factors3.dateff)
    ff_factors3.set_index('date', inplace=True)

    ##############################
    # Compare results
    ##############################
    # To compare from 1970.
    # ff_factors1 = ff_factors1['1970':]
    # ff_factors2 = ff_factors2['1970':]
    # ff_factors3 = ff_factors3['1970':]
    # ff_nfirms1 = ff_nfirms1['1970':]
    # ff_nfirms2 = ff_nfirms2['1970':]

    print('\nCompare pyanomaly with WRDS')
    print('factors')
    ff_factors12 = compare_data(ff_factors1, ff_factors2, on='date', tolerance=0.1)
    print('\nnumber of firms')
    ff_nfirms12 = compare_data(ff_nfirms1, ff_nfirms2, on='date', tolerance=0.1)
    ff_factors12[['smb_x', 'smb_y']].plot()
    plt.show()

    print('\nCompare pyanomaly with FF')
    ff_factors13 = compare_data(ff_factors1, ff_factors3, on='date', tolerance=0.1)
    ff_factors13[['smb_x', 'smb_y']].plot()
    plt.show()

    print('\nCompare WRDS with FF')
    ff_factors23 = compare_data(ff_factors2, ff_factors3, on='date', tolerance=0.1)
    ff_factors23[['smb_x', 'smb_y']].plot()
    plt.show()
