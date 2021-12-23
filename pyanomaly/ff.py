import numpy as np
import pandas as pd
import statsmodels.api as sm

from .globals import *
from .datatools import *
from .analytics import classify, two_dim_sort
from .characteristics import FUNDA, CRSPM
from .wrdsdata import WRDS



def make_ff_factors():
    """Generate fama-french factors using the Fama and French (1993) method.

    This code follows the code in wrds (WRDS hereafter) but maintains the overall architecture of pyanomaly, which results in differences.
    wrds code: there are minor difference (https://wrds-www.wharton.upenn.edu/pages/support/applications/python-replications/fama-french-factors-python/

    1. primary stocks:
    - WRDS choose a stock with the max me, whereas we use linkprim in ccmxpf_linktable and trading volume.
      For details of primary stock identification, see wrds.add_gvkey_to_crsp()
    2. crsp-comp merge:
    - We use see wrds.add_gvkey_to_crsp() to merge. This results in slightly a different merge result.
    3. When fiscal year changes, there are dups in funda. We use the last record while WRDS uses the first one.
    4. I apply filtering conditions such as shrcd in (10, 11) at the beginning, which aslo cuases slight differences.
    5. Pyanomaly convention is that date is the day characteristics are calculated and return is over the next month.
       However, in this function, I follow the usual convention, ie, return is calculated onver the month of date.

    Returns:
        ff_factors: index: date, columns: BH, BM, BL, SH, SM, SL, hml, smb
        ff_nfirms: index: date, columns: BH, BM, BL, SH, SM, SL, hml, smb
    """

    # CRSPM
    crspm = CRSPM()
    crspm.load_data()
    crspm.preprocess()

    cm = crspm.data.reset_index()
    cm = cm[cm.primary]  # primary security only
    cm['fyear'] = (cm.date - pd.DateOffset(months=6)).dt.year
    cm['me'] = cm.me_company  # Use company level me.
    cm['me_1'] = cm.groupby('permno').me.shift(1)

    # Filter crspm
    # common shares of domestic firms and me > 0
    cm = cm[cm.shrcd.isin([10, 11]) & (cm.me > 0)]

    # FUNDA
    funda = FUNDA()
    funda.load_data()
    funda.preprocess()
    fa = funda.data.reset_index()
    fa['fyear'] = fa.datadate.dt.year  # Cannot use funda.fyear as it is datadate.year if month > 6, year-1 otherwise.
    fa = fa.drop_duplicates(['gvkey', 'fyear'], keep='last')  # Fiscal year change can cause dup. Keep the most recent.
    fa['rcount'] = fa.groupby('gvkey').cumcount()  # reset rcount (since some rows are dropped)
    fa = fa[['gvkey', 'datadate', 'be_', 'rcount', 'fyear']]

    # Filter funda
    fa = fa[(fa.rcount >= 1)]
    fa = fa[fa.be_ > 0]

    # ME for BEME and SIZE
    # BEME: t-1 year-end me, SIZE: t year June me.
    me_jun = cm.loc[cm.date.dt.month == 6, ['permno', 'fyear', 'me', 'exchcd', 'gvkey']].rename(columns={'me': 'me_jun'})
    me_dec = cm.loc[cm.date.dt.month == 12, ['permno', 'fyear', 'me']].rename(columns={'me': 'me_dec'})
    me = pd.merge(me_dec, me_jun, on=['fyear', 'permno'])

    # Merge funda with me
    fa = pd.merge(fa, me, on=['fyear', 'gvkey'])
    fa['be_me'] = fa.be_ / fa.me_dec
    fa = fa.set_index(['fyear', 'permno'])

    # SIZE
    size_col = 'me_jun'
    size_split = [0.5, 1.0]
    size_labels = ['s', 'b']
    size_by = size_col + '_by'
    fa[size_by] = fa[size_col]
    fa.loc[(fa.exchcd != 1) | (fa.rcount < 2), size_by] = None  # NYSE and at least 2 years in funda
    fa['size_class'] = classify(fa, size_col, size_split, ascending=True, by=size_by, labels=size_labels)

    # BM
    bm_col = 'be_me'
    bm_split = [0.3, 0.7, 1.0]
    bm_labels = ['h', 'm', 'l']
    bm_by = bm_col + '_by'
    fa[bm_by] = fa[bm_col]
    fa.loc[(fa.exchcd != 1) | (fa.rcount < 2), bm_by] = None  # NYSE and at least 2 years in funda
    fa['bm_class'] = classify(fa, bm_col, bm_split, ascending=False, by=bm_by, labels=bm_labels)

    # Remove unclassified rows. This happens if there are no NYSE stocks with at least 2 years in funda.
    fa = fa[fa['size_class'].isin(size_labels) & fa['bm_class'].isin(bm_labels)]

    # Merge annual fa with monthly cm
    cm['fyear'] -= 1
    cm = pd.merge(cm, fa[fa.columns.difference(cm.columns)], on=['permno', 'fyear'])
    cm = cm.set_index(['date', 'permno'])

    # Double sort on size and bm.
    ff_factors = two_dim_sort(cm, 'size_class', 'bm_class', 'ret', weight_col='me_1')

    ff_factors = ff_factors.reset_index(level=[-2, -1])  # index: date/size/bm -> date
    ff_factors['class'] = ff_factors['size_class'] + ff_factors['bm_class']
    ff_factors = ff_factors.pivot(columns='class', values='ret')

    # smb and hml factors
    ff_factors['hml'] = (ff_factors['bh'] + ff_factors['sh']) / 2 - \
                        (ff_factors['bl'] + ff_factors['sl']) / 2
    ff_factors['smb'] = (ff_factors['sh'] + ff_factors['sm'] + ff_factors['sl']) / 3 - \
                        (ff_factors['bh'] + ff_factors['bm'] + ff_factors['bl']) / 3
    # ff_factors['hml'] = (ff_factors['BHL'] + ff_factors['SHL']) / 2
    # ff_factors['smb'] = (ff_factors['SBH'] + ff_factors['SBM'] + ff_factors['SBL']) / 3

    # Firm count
    ff_nfirms = two_dim_sort(cm, 'size_class', 'bm_class', 'ret', function='count')
    ff_nfirms = ff_nfirms.reset_index(level=[-2, -1])  # index: date/size/bm -> date
    ff_nfirms['class'] = ff_nfirms['size_class'] + ff_nfirms['bm_class']
    ff_nfirms = ff_nfirms.pivot(columns='class', values='ret')

    ff_nfirms['hml'] = ff_nfirms['bh'] + ff_nfirms['sh'] + ff_nfirms['bl'] + ff_nfirms['sl']
    ff_nfirms['smb'] = ff_nfirms['sh'] + ff_nfirms['sm'] + ff_nfirms['sl'] + \
                       ff_nfirms['bh'] + ff_nfirms['bm'] + ff_nfirms['bl']

    return cm, ff_factors, ff_nfirms


def test_ff_factors():
    cm, ff_factors, ff_nfirms = make_ff_factors()

    ff_factors2 = WRDS.read_data('factors_monthly')
    ff_factors2['date'] = pd.to_datetime(ff_factors2.dateff)
    ff_factors = pd.merge(ff_factors, ff_factors2, on='date')

    compare_data(ff_factors)

    return ff_factors, ff_nfirms


if __name__ == '__main__':
    os.chdir('../')

    ff_factors, ff_nfirms = test_ff_factors()

