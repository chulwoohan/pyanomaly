"""This module defines classes for firm characteristic generation.

    * `FUNDA`
        Class to generate firm characteristics from funda.
    * `FUNDQ`
        Class to generate firm characteristics from fundq.
    * `CRSPM`
        Class to generate firm characteristics from crspm.
    * `CRSPD`
        Class to generate firm characteristics from crspd.
    * `Merge`
        Class to generate firm characteristics from a merged dataset of funda, fundq, crspm, and crspd.
"""

import numpy as np
import pandas as pd

from pyanomaly.globals import *
from pyanomaly.wrdsdata import WRDS
from pyanomaly.panel import *
from pyanomaly.datatools import *
from pyanomaly.analytics import *
from pyanomaly.fileio import read_from_file
from pyanomaly.numba_support import *
# from pyanomaly.multiprocess import multiprocess


################################################################################
#
# FUNDA
#
################################################################################

class FUNDA(Panel):
    """Class to generate firm characteristics from funda.

    The firm characteristics generated in this class can be viewed using ``FUNDA.show_available_functions()``:

    >>> FUNDA().show_available_functions()

    Refer to the manual for usage.

    Args:
        alias: Characteristic column name in ``mapping.xlsx``. If None, function names (without 'c\_') are used as the
            characteristic names.
        data: DataFrame with index = date/id, sorted on id/date.
        freq: Frequency of `data`. ANNUAL, QUARTERLY, MONTHLY, or DAILY. Default to ANNUAL.
    """

    def __init__(self, alias=None, data=None, freq=ANNUAL):
        super().__init__(alias, data, freq, ANNUAL)

    def load_data(self, sdate=None, edate=None):
        """Load funda data from file.

        The loaded data is sorted on gvkey/datadate, has index = datadate/gvkey, and stored in the `data` attribute.
        """

        elapsed_time(f'Loading funda...')

        data = WRDS.read_data('funda')
        if sdate:
            data = data[data.datadate >= sdate]
        if edate:
            data = data[data.datadate <= edate]

        self.data = data.set_index(['datadate', 'gvkey']).sort_index(level=['gvkey', 'datadate'])

        elapsed_time('funda loaded.')
        self.inspect_data()

    def convert_currency(self):
        """Convert the currency of `data` to USD.

        This method needs to be called if

            i) the data contains non USD-denominated firms, e.g., CAD, and
            ii) CRSP's market equity is used, which is always in USD.
        """

        if (self.data['curcd'] != 'USD').any():
            self.data = WRDS.convert_fund_currency_to_usd(self.data, table='funda')

    def convert_to_monthly(self, lag=4, limit=12):
        """Populate `data` to monthly frequency.

        The index of `data` changes from datadate/gvkey to date/gvkey and datadate is kept as a column.
        The date and datadate has a gap of at least `lag` months.

        Args:
            lag: Minimum months between date and datadate: `lag` = 4 assumes funda is available 4 months after datadate.
            limit: Maximum months to forward-fill the data.
        """

        elapsed_time(f'Populating funda data to monthly frequency...')

        # Populate data
        fa = self.data.reset_index()
        fa['date'] = fa['datadate']
        fa = fa.set_index(['date', 'gvkey'])
        fa = populate(fa, MONTHLY, limit=limit)

        # Shift data by 'lag' months.
        ids = fa.index.get_level_values('gvkey')
        fa = fa.shift(lag)
        fa = fa[ids == np.roll(ids, lag)]
        # fa['rcount'] = fa.groupby('gvkey').cumcount()
        # fa = fa[fa.rcount >= lag]  # delete first 'lag' months.
        self.data = fa
        self.freq = MONTHLY
        elapsed_time(f'funda data populated.')

    def merge_with_fundq(self, fundq):
        """Merge funda with fundq.

        If variable X is available in fundq, i.e., Xq or Xy exists, it is used, otherwise,
        X in funda is used. Xq(y) replaces X if:

            i) X is missing, or
            ii) Xq(y) is not missing and fundq.datadate > funda.datadate.

        NOTE:
            JKP create characteristics in funda and fundq separately and merge them, whereas we merge the raw data
            first and then calculate characteristics. Since some variables in funda are not available in fundq, eg, ebitda,
            JKP make those unavailable variables from other variables and create characteristics, even when they are
            available in funda. We prefer to merge funda with fundq at the raw data level and create characteristics from
            the merged data.

        Columns in both funda and fundq:

            datadate, cusip, cik, sic, naics, sale, revt, cogs, xsga, dp, xrd, ib, nopi, spi, pi, txp, ni, txt, xint,
            capx, oancf, gdwlia, gdwlip, rect, act, che, ppegt, invt, at, aco, intan, ao, ppent, gdwl, lct, dlc, dltt,
            lt, pstk, ap, lco, lo, drc, drlt, txdi, ceq, scstkc, csho, prcc_f, oibdp, oiadp, mii, xopr, xi, do, xido,
            ibc, dpc, xidoc, fincf, fiao, txbcof, dltr, dlcch, prstkc, sstk, dv, ivao, ivst, re, txditc, txdb, seq,
            mib, icapt, ajex, curcd, exratd, rcount

        Columns in funda but not in fundq:

            xad, gp, ebitda, ebit, txfed, txfo, dvt, ob, gwo, fatb, fatl, dm, dcvt, cshrc, dcpstk, emp, xlr, ds, dvc,
            itcb, pstkrv, pstkl, dltis, ppenb, ppenls

        Args:
            fundq: FUNDQ instance.
        """

        elapsed_time('Merging funda with fundq...')

        fa = self.data
        fq = fundq.generate_funda_vars()  # quarterly updated annual data

        if REPLICATE_JKP:
            for col in fa.columns:
                if col not in fq.columns:
                    fq[col] = np.nan

        common_columns = list(fq.columns.intersection(fa.columns))  # columns in both funda and fundq
        common_columns_q = [col + '_q' for col in common_columns]
        fq = fq[common_columns]

        # Merge funda with fundq. fundq.X changes to fundq.X_q
        fa = fa.merge(fq, on=['date', 'gvkey'], how='outer', suffixes=['', '_q'])
        fa.sort_index(level=['gvkey', 'date'], inplace=True)

        # Replace X with X_q
        for col, qcol in zip(common_columns, common_columns_q):
            if col == 'datadate':
                continue
            if REPLICATE_JKP:
                cond = fa['datadate'].isna() | (fa['datadate'] < fa['datadate_q'])
            else:
                cond = fa[col].isna() | (~fa[qcol].isna() & (fa['datadate'] < fa['datadate_q']))
            fa[col] = np.where(cond, fa[qcol], fa[col])

        cond = fa['datadate'].isna() | (fa['datadate'] < fa['datadate_q'])
        fa['datadate'] = np.where(cond, fa['datadate_q'], fa['datadate'])

        fa.drop(columns=common_columns_q, inplace=True)

        self.data = fa
        elapsed_time('funda and fundq merged.')

    def add_crsp_me(self, crspm, method='latest'):
        """Replace funda's market equity ('me') with crspm's firm level market equity ('me_company').

        This method also adds a column, 'me_fiscal' (me_company on datadate), to `data`.
        Without calling this function, me and me_fiscal are set to me of funda (prcc_f * csho).

        Args:
            crspm: CRSPM instance.
            method: How to merge crsp me with funda. 'latest': latest me; 'year_end': December me.
        """

        elapsed_time('Adding crspm me to funda...')

        cm = crspm.data
        cm = cm.loc[(cm.primary == True) & (cm.me_company > 0), ['gvkey', 'me_company']]
        cm = cm.rename(columns={'me_company': 'me'})

        fa = self.data
        drop_columns(fa, ['me', 'me_fiscal'])  # remove me from funda

        if method == 'latest':
            md = fa.merge(cm, on=['date', 'gvkey'], how='left')

            # Add me_fiscal
            md = md.reset_index()
            tmp = md[['gvkey', 'date', 'me']].rename(columns={'date': 'datadate', 'me': 'me_fiscal'})
            md = md.merge(tmp, on=['gvkey', 'datadate'], how='left')
            md = md.set_index(['date', 'gvkey'])
        elif method == 'year_end':
            cm = cm.reset_index()
            cm = cm[cm.date.dt.month == 12]
            cm['year'] = cm.date.dt.year
            drop_columns(cm, ['date', 'permno'])
            fa = fa.reset_index()
            fa['year'] = fa.datadate.dt.year
            md = fa.merge(cm, on=['year', 'gvkey'], how='left').set_index(['date', 'gvkey'])
            del md['year']
        else:
            raise ValueError(f"Methods to merge crsp me with funda: ['latest', 'year_end']. '{method}' is given.")

        self.data = md
        elapsed_time(f'crspm me added to funda. funda shape: {md.shape}')


    def update_variables(self):
        """Preprocess data before creating characteristics.

        1. Replace missing values with other variables.
        2. Create frequently used variables.
        """

        elapsed_time(f'updating funda variables...')
        self.add_row_count()  # add row count (rcount).

        fa = self.data

        # Cleansing
        # non-negative variables with a negative value -> set to nan.
        for var in ['at', 'sale', 'revt', 'dv', 'che']:
            fa.loc[fa[var] < 0, var] = np.nan  # should be >= 0.

        # tiny values -> set to 0.
        for var in fa.columns:
            if fa[var].dtype == float:
                fa[var] = np.where(is_zero(fa[var]), 0, fa[var])

        fa['prcc_f'] = fa.prcc_f.abs()
        fa.rename(columns={'at': 'at_', 'lt': 'lt_'}, inplace=True)  # 'at', 'lt' are reserved names in python.

        # Income statement
        fa['sale'] = fa.sale.fillna(fa.revt)
        fa['gp'] = fa.gp.fillna(fa.sale - fa.cogs)  # revt - cogs in HXZ
        fa['xopr'] = fa.xopr.fillna(fa.cogs + fa.xsga)
        fa['ebitda'] = fa.ebitda.fillna(fa.oibdp.fillna((fa.sale - fa.xopr).fillna(fa.gp - fa.xsga)))
        fa['ebit'] = fa.ebit.fillna(fa.oiadp.fillna(fa.ebitda - fa.dp))
        fa['pi'] = fa.pi.fillna(fa.ebit - fa.xint + fa.spi.fillna(0) + fa.nopi.fillna(0))
        fa['xido'] = fa.xido.fillna(fa.xi + fa.do.fillna(0))
        fa['ib'] = fa.ib.fillna((fa.ni - fa.xido).fillna(fa.pi - fa.txt - fa.mii.fillna(0)))
        fa['nix'] = fa.ni.fillna(fa.ib + (fa.xido.fillna(0)).fillna(fa.xi + fa.do))
        fa['dvt'] = fa.dvt.fillna(fa.dv)

        # Balance sheet - Assets
        fa['pstkrv'] = fa.pstkrv.fillna(fa.pstkl.fillna(fa.pstk))
        fa['seq'] = fa.seq.fillna((fa.ceq + fa.pstkrv.fillna(0)).fillna(fa.at_ - fa.lt_))
        fa['at_'] = fa.at_.fillna(fa.seq + fa.dltt + fa.lct.fillna(0) + fa.lo.fillna(0) + fa.txditc.fillna(
            0))  # used in seq before this conversion
        fa.loc[fa.at_ < ZERO, 'at_'] = np.nan  # asset should be > 0
        fa['avgat'] = self.rolling(fa.at_, 2, 'mean')  # 2-year average at.
        fa['act'] = fa.act.fillna(fa.rect + fa.invt + fa.che + fa.aco)

        # Balance sheet - Liabilities
        fa['lct'] = fa.lct.fillna(fa.ap + fa.dlc + fa.txp + fa.lco)  # used in 'at' before this conversion
        fa['txditc'] = fa.txditc.fillna(self.sum(fa.txdb, fa.itcb))  # used in 'at' before this conversion

        # Balance sheet - Financing
        fa['debt'] = self.sum(fa.dltt, fa.dlc)
        fa['be'] = fa.seq + fa.txditc.fillna(0) - fa.pstkrv.fillna(0)
        fa['bev'] = (fa.icapt + fa.dlc.fillna(0) - fa.che.fillna(0)).fillna(
            fa.seq + fa.debt - fa.che.fillna(0) + fa.mib.fillna(0))

        # Market equities
        fa['me'] = fa.prcc_f * fa.csho
        fa['me_fiscal'] = fa['me']

        # Accruals
        cowc = fa.act - fa.che - (fa.lct - fa.dlc.fillna(0))
        nncoa = fa.at_ - fa.act - fa.ivao.fillna(0) - (fa.lt_ - fa.lct - fa.dltt)
        fa['oacc'] = (fa.ib - fa.oancf).fillna(self.diff(cowc + nncoa))
        fa.loc[(fa.ib.isna() | fa.oancf.isna()) & (fa.rcount < 12), 'oacc'] = np.nan  # if diff(cowc + nncoa) is used, the
                                                                                   # first row should be set to nan.
        fa['ocf'] = fa.oancf.fillna((fa.ib - fa.oacc).fillna(fa.ib + fa.dp))

        # for some characteristics in GHZ
        fa['acc_'] = self.diff(fa.act - fa.che - fa.lct + fa.dlc + fa.txp) + fa.dp
        fa['sic2'] = fa.sic.str[:2]  # first 2-digit of sic.

        # non-negative variables with a negative value -> set to nan.
        for var in ['be', 'bev', 'debt', 'seq']:
            fa.loc[fa[var] < 0, var] = np.nan  # should be >= 0.

        self.data = fa
        elapsed_time('funda variables updated.')

    def postprocess(self):
        """Postprocess data.

        This method deletes temporary variables (variables starting with '\_') and replaces infinite values with nan.
        This method can be overridden to add code to trim or winsorize characteristics.
        """

        elapsed_time('Postprocessing funda...')
        fa = self.data

        # Drop temporary columns that start with '_'
        del_cols = [col for col in fa if col[0] == '_']
        drop_columns(fa, del_cols)

        columns = fa.columns.intersection(self.get_available_chars())
        inspect_data(fa[columns], option=['summary', 'nans'])

        log('Replacing inf with nan...')
        fa.replace([np.inf, -np.inf], np.nan, inplace=True)

        elapsed_time('funda postprocessed.')

    #######################################
    # Characteristics
    #######################################
    def c_at_gr1(self):
        """Asset growth. Cooper, Gulen, and Schill (2008)"""

        fa = self.data
        char = self.pct_change(fa.at_)
        return char

    def c_sale_gr1(self):
        """Annual sales growth. Lakonishok, Shleifer, and Vishny (1994)"""

        fa = self.data
        char = self.pct_change(fa.sale)
        return char

    def c_sale_gr3(self):
        """Three-year sales growth. Lakonishok, Shleifer, and Vishny (1994)"""

        fa = self.data
        char = self.pct_change(fa.sale, 3)
        return char

    def c_debt_gr3(self):
        """Composite debt issuance. Lyandres, Sun, and Zhang (2008)"""

        fa = self.data
        if REPLICATE_JKP:
            char = self.pct_change(fa.debt, 3)  # JKP
        else:
            char = np.log(fa.debt / self.shift(fa.debt, 5))
        return char

    def c_inv_gr1a(self):
        """Inventory change. Thomas and Zhang (2002)"""

        fa = self.data
        if REPLICATE_JKP:
            char = self.diff(fa.invt) / fa.at_  # JKP use at at t.
        else:
            char = self.diff(fa.invt) / fa.avgat  # JKP use at at t.
        return char

    def c_lti_gr1a(self):
        """Chagne in long-term investments. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.ivao) / fa.at_
        return char

    def c_coa_gr1a(self):
        """Change in current operating assets. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.act - fa.che) / fa.at_
        return char

    def c_col_gr1a(self):
        """Change in current Ooperating liabilities. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.lct - fa.dlc.fillna(0)) / fa.at_
        return char

    def c_cowc_gr1a(self):
        """Change in net non-cash working capital. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.act - fa.che - fa.lct + fa.dlc.fillna(0)) / fa.at_
        return char

    def c_ncoa_gr1a(self):
        """Change in non-current operating assets. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.at_ - fa.act - fa.ivao.fillna(0)) / fa.at_
        return char

    def c_ncol_gr1a(self):
        """Change in non-current operating liabilities. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.lt_ - fa.lct - fa.dltt) / fa.at_
        return char

    def c_nncoa_gr1a(self):
        """Change in net non-current operating assets. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.at_ - fa.act - fa.ivao.fillna(0) - (fa.lt_ - fa.lct - fa.dltt)) / fa.at_
        return char

    def c_noa_gr1a(self):
        """Change in net operating assets. Hirshleifer et al. (2004)"""

        fa = self.data
        noa = fa.act - fa.che + fa.at_ - fa.act - fa.ivao.fillna(0) - (
                    fa.lct - fa.dlc.fillna(0) + (fa.lt_ - fa.lct - fa.dltt))
        char = self.diff(noa) / self.shift(fa.at_)
        return char

    def c_fnl_gr1a(self):
        """Change in financial liabilities. Richardson et al. (2005)"""

        fa = self.data
        fnl = fa.debt + fa.pstkrv
        if REPLICATE_JKP:
            char = self.diff(fnl) / fa.at_  # jkp uses 'at' at t while hxz uses 'at' at t-1.
        else:
            char = self.diff(fnl) / fa.avgat  # jkp uses 'at' at t while hxz uses 'at' at t-1.
        return char

    def c_nfna_gr1a(self):
        """Change in net financial assets. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.ivst.fillna(0) + fa.ivao.fillna(0) - (fa.debt + fa.pstkrv)) / fa.at_
        return char

    # def c_tax_gr1a(self):
    #     # Tax expense surprise (JKP). Thomas and Zhang (2011)
    #     => defined in fundq as chtx following the original definition
    #     fa = self.data
    #     char = self.diff(fa.txt) / fa.at_
    #     return char

    def c_ebit_sale(self):
        """Profit margin. Soliman (2008)"""

        fa = self.data
        char = fa.ebit / fa.sale
        char[fa.sale < ZERO] = np.nan
        return char

    def c_gp_at(self):
        """Gross profits-to-assets. Novy-Marx (2013)"""

        fa = self.data
        return fa.gp / fa.at_

    def c_cop_at(self):
        """Cash-based operating profitablility. Ball et al. (2016)"""

        fa = self.data
        cop = fa.ebitda + fa.xrd.fillna(0) - fa.oacc
        char = cop / fa.at_
        return char

    def c_ni_be(self):
        """Return on equity. Haugen and Baker (1996)"""

        fa = self.data
        char = fa.ib / fa.be
        char[fa.be == 0] = np.nan
        return char

    def c_ebit_bev(self):
        """Return on net operating assets. Soliman (2008)"""

        fa = self.data
        char = fa.ebit / fa.bev
        char[fa.bev == 0] = np.nan
        return char

    def c_netis_at(self):
        """Net external finance. Bradshaw, Richardson, and Sloan (2006)"""

        # nxf = nef + ndf
        # nef = sstk - prstkc - dv, ndf = dltis - dltr + dlcch
        fa = self.data
        ddltt = self.diff(fa.dltt)
        ddlc = self.diff(fa.dlc)
        dbnetis = self.sum(self.sum(fa.dltis, -fa.dltr).fillna(ddltt), fa.dlcch.fillna(ddlc))
        if REPLICATE_JKP:
            netis = self.sum(fa.sstk, -fa.prstkc) + dbnetis
        else:
            netis = self.sum(fa.sstk, -fa.prstkc, -fa.dv) + dbnetis
        return netis / fa.at_

    def c_eqnetis_at(self):
        """Net equity finance. Bradshaw, Richardson, and Sloan (2006)"""

        fa = self.data
        if REPLICATE_JKP:
            return self.sum(fa.sstk, -fa.prstkc) / fa.at_
        else:
            return self.sum(fa.sstk, -fa.prstkc, -fa.dv) / fa.at_

    def c_dbnetis_at(self):
        """Net debt finance. Bradshaw, Richardson, and Sloan (2006)"""

        fa = self.data
        ddltt = self.diff(fa.dltt)
        ddlc = self.diff(fa.dlc)
        dbnetis = self.sum(self.sum(fa.dltis, -fa.dltr).fillna(ddltt), fa.dlcch.fillna(ddlc))
        return dbnetis / fa.at_

    def c_oaccruals_at(self):
        """Operating Accruals (JKP). Sloan (1996)"""

        fa = self.data
        if REPLICATE_JKP:
            char = fa.oacc / fa.at_  # JKP use at in the denominator, but Sloan uses the average.
            return char
        else:
            char = fa.oacc / fa.avgat  # JKP use at in the denominator, but Sloan uses the average.
            return char

    def c_acc(self):
        """Operating accruals (GHZ, Org). Sloan (1996)"""

        fa = self.data
        char = (fa.ib - fa.oancf).fillna(fa.acc_) / fa.avgat
        return char

    def c_oaccruals_ni(self):
        """Percent Operating Accruals (JKP). Hafzalla, Lundholm, and Van Winkle (2011)"""

        fa = self.data
        char = fa.oacc / fa.nix.abs()
        char[fa.nix == 0] = np.nan
        return char

    def c_pctacc(self):
        """Percent operating accruals (GHZ, Org). Hafzalla, Lundholm, and Van Winkle (2011)"""

        fa = self.data
        denom = fa.ib.abs()
        denom[denom == 0] = 0.01
        char = (fa.ib - fa.oancf).fillna(fa.acc_) / denom
        return char

    def c_taccruals_at(self):
        """Total Accruals. Richardson et al. (2005)"""

        fa = self.data
        tacc = fa.oacc + self.diff(fa.ivst.fillna(0) + fa.ivao.fillna(0) - (fa.debt + fa.pstkrv))
        char = tacc / fa.at_
        return char

    def c_taccruals_ni(self):
        """Percent total accruals. Hafzalla, Lundholm, and Van Winkle (2011)"""

        fa = self.data
        tacc = fa.oacc + self.diff(fa.ivst.fillna(0) + fa.ivao.fillna(0) - (fa.debt + fa.pstkrv))
        char = tacc / fa.nix.abs()
        char[fa.nix == 0] = np.nan
        return char

    def c_noa_at(self):
        """Net operating assets. Hirshleifer et al. (2004)"""

        fa = self.data
        noa = fa.act - fa.che + fa.at_ - fa.act - fa.ivao.fillna(0) - (
                    fa.lct - fa.dlc.fillna(0) + (fa.lt_ - fa.lct - fa.dltt))
        char = noa / self.shift(fa.at_)
        return char

    def c_opex_at(self):
        """Operating leverage. Novy-Marx (2011)"""

        fa = self.data
        return fa.xopr / fa.at_  # HXZ: cogs + xsga

    def c_at_turnover(self):
        """Capital turnover. Haugen and Baker (1996)"""

        fa = self.data
        char = fa.sale / fa.avgat
        return char

    def c_sale_bev(self):
        """Asset turnover. Soliman (2008)"""

        fa = self.data
        char = fa.sale / fa.bev
        char[fa.bev == 0] = np.nan
        return char

    def c_rd_sale(self):
        """R&D to sales. Chan, Lakonishok, and Sougiannis (2001) (Guo, Lev, and Shi (2006) in GHZ)"""

        fa = self.data
        char = fa.xrd / fa.sale
        char[fa.sale == 0] = np.nan
        return char

    def c_be_me(self):
        """Book-to-market (December ME). Rosenberg, Reid, and Lanstein (1985)"""

        fa = self.data
        # return fa.ceq / fa.me  # GHZ
        return fa.be / fa.me

    def c_at_me(self):
        """Assets-to-market. Fama and French (1992)"""

        fa = self.data
        return fa.at_ / fa.me

    def c_ni_me(self):
        """Earnings to price. Basu (1983)"""

        fa = self.data
        return fa.ib / fa.me

    def c_sale_me(self):
        """Sales to price. Barbee, Mukherji, and Raines (1996)"""

        fa = self.data
        return fa.sale / fa.me

    def c_ocf_me(self):
        """Operating Cash flows to price (JKP). Desai, Rajgopal, and Venkatachalam (2004)"""

        fa = self.data
        char = fa.ocf / fa.me
        return char

    def c_cfp(self):
        """Operating Cash flows to price (Org, GHZ). Desai, Rajgopal, and Venkatachalam (2004)"""

        fa = self.data
        char = fa.oancf.fillna(fa.ib - fa.acc_) / fa.me
        return char

    def c_fcf_me(self):
        """Cash flow-to-price. Lakonishok, Shleifer, and Vishny (1994)"""

        fa = self.data
        char = (fa.ocf - fa.capx) / fa.me
        return char

    def c_rd_me(self):
        """R&D to market. Chan, Lakonishok, and Sougiannis (Guo, Lev, and Shi (2006) in GHZ)"""

        fa = self.data
        return fa.xrd / fa.me

    def c_bev_mev(self):
        """Book-to-market enterprise value. Penman, Richardson, and Tuna (2007)"""

        fa = self.data
        mev = fa.me + fa.debt - fa.che.fillna(0)
        mev[mev < ZERO] = np.nan
        return fa.bev / mev

    def c_eqpo_me(self):
        """Payout yield. Boudoukh et al. (2007)"""

        fa = self.data
        return (fa.dvt + fa.prstkc) / fa.me

    def c_eqnpo_me(self):
        """Net payout yield. Boudoukh et al. (2007)"""

        fa = self.data
        return (fa.dvt - self.sum(fa.sstk, -fa.prstkc)) / fa.me

    def c_ebitda_mev(self):
        """Enterprise multiple. Loughran and Wellman (2011)"""

        fa = self.data
        mev = fa.me + fa.debt - fa.che.fillna(0)
        mev[mev < ZERO] = np.nan

        if REPLICATE_JKP:
            return fa.ebitda / mev  # in JKP. HXZ add pstkrv in the enterprise value.
        else:
            return mev / fa.ebitda  # This is the correct definition of enterprise multiple.

    def c_ocf_at(self):
        """Operating cash flow to assets. Bouchard et al. (2019)"""

        fa = self.data
        char = fa.ocf / fa.at_
        return char

    def c_ocf_at_chg1(self):
        """Change in operating cash flow to assets. Bouchard et al. (2019)"""

        fa = self.data
        self.prepare(['ocf_at'])
        char = self.diff(fa.ocf_at)
        return char

    def c_cash_at(self):
        """Cash-to-assets. Palazzo (2012)"""

        fa = self.data
        return fa.che / fa.at_

    def c_ppeinv_gr1a(self):
        """Changes in PPE and inventory/assets. Lyandres, Sun, and Zhang (2008)"""

        fa = self.data
        char = self.diff(fa.ppegt + fa.invt) / self.shift(fa.at_)
        return char

    def c_lnoa_gr1a(self):
        """Change in long-term net operating assets. Fairfield, Whisenant, and Yohn (2003)"""

        fa = self.data
        if REPLICATE_JKP:
            lnoa = fa.ppent + fa.intan + fa.ao - fa.lo + fa.dp
            char = self.diff(lnoa) / (fa.at_ + self.shift(fa.at_))  # kelly definition
        else:
            char = (self.diff(fa.ppent + fa.intan + fa.ao - fa.lo) + fa.dp) / fa.avgat
        return char

    def c_capx_gr1(self):
        """CAPEX growth (1 year). Xie (2001)"""

        fa = self.data
        char = self.pct_change(fa.capx)
        return char

    def c_capx_gr2(self):
        """Two-year investment growth. Anderson and Garcia-Feijoo (2006)"""

        fa = self.data
        char = self.pct_change(fa.capx, 2)
        return char

    def c_capx_gr3(self):
        """Three-year investment growth. Anderson and Garcia-Feijoo (2006)"""

        fa = self.data
        char = self.pct_change(fa.capx, 3)
        return char

    def c_sti_gr1a(self):
        """Change in short-term investments. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.ivst) / fa.at_
        # char[fa.at_ < ZERO] = np.nan
        return char

    def c_rd5_at(self):
        """R&D capital-to-assets. Li (2011)"""

        fa = self.data
        char = (fa.xrd + 0.8 * self.shift(fa.xrd) + 0.6 * self.shift(fa.xrd, 2) +
                0.4 * self.shift(fa.xrd, 3) + 0.2 * self.shift(fa.xrd, 4)) / fa.at_
        return char

    def c_age(self):
        """Firm age. Jiang, Lee, and Zhang (2005)"""

        fa = self.data
        return (fa.rcount + 1) * 12 / self.freq

    def _chg_to_exp(self, char):
        if REPLICATE_JKP:
            exp = self.rolling(char, 2, 'mean', min_n=2, lag=1)  # JKP require both lagged values.
        else:
            exp = self.rolling(char, 2, 'mean', min_n=1, lag=1)  # min_n=1: calculate if there is at least one observation.
        exp[exp < ZERO] = np.nan
        return char / exp - 1

    def c_dsale_dinv(self):
        """Sales growth to inventory growth. Abarbanell and Bushee (1998)"""

        fa = self.data
        char = self._chg_to_exp(fa.sale) - self._chg_to_exp(fa.invt)
        return char

    def c_dsale_drec(self):
        """Sales growth to receivable growth. Abarbanell and Bushee (1998)"""

        fa = self.data
        char = self._chg_to_exp(fa.sale) - self._chg_to_exp(fa.rect)
        return char

    def c_dgp_dsale(self):
        """Gross margin growth to sales growth. Abarbanell and Bushee (1998)"""

        fa = self.data
        char = self._chg_to_exp(fa.gp) - self._chg_to_exp(fa.sale)  # Original definition uses (sale-cogs),
                                                                    # but jkp gives priority to gp.
        return char

    def c_dsale_dsga(self):
        """Sales growth to SG&A growth. Abarbanell and Bushee (1998)"""

        fa = self.data
        char = self._chg_to_exp(fa.sale) - self._chg_to_exp(fa.xsga)
        return char

    def c_debt_me(self):
        """Debt to market. Bhandari (1988)"""

        fa = self.data
        # return fa.lt_ / fa.me  # GHZ
        return fa.debt / fa.me

    def c_netdebt_me(self):
        """Net debt-to-price. Penman, Richardson, and Tuna (2007)"""

        fa = self.data
        return (fa.debt - fa.che.fillna(0)) / fa.me

    def c_capex_abn(self):
        """Abnormal corporate investment. Titman, Wei, and Xie (2004)"""

        fa = self.data
        capx_sale = fa.capx / fa.sale
        capx_sale[fa.sale < ZERO] = np.nan
        denom = self.rolling(capx_sale, 3, 'mean', lag=1)
        char = capx_sale / denom - 1
        char[denom < ZERO] = np.nan
        return char

    def c_inv_gr1(self):
        """Inventory growth. Belo and Lin (2012)"""

        fa = self.data
        char = self.pct_change(fa.invt)
        return char

    def c_be_gr1a(self):
        """Chage in common equity. Richardson et al. (2005)"""

        fa = self.data
        # char = self.diff(fa.ceq) / fa.avgat  # original definition
        if REPLICATE_JKP:
            char = self.diff(fa.be) / fa.at_  # jkp
        else:
            char = self.diff(fa.be) / fa.avgat
        return char

    def c_op_at(self):
        """Operating profits-to-assets. Ball et al. (2016)"""

        fa = self.data
        op = fa.ebitda + fa.xrd.fillna(0)
        return op / fa.at_

    def c_pi_nix(self):
        """Taxable income to income (JKP). Lev and Nissim (2004)"""

        fa = self.data
        char = fa.pi / fa.nix
        char[(fa.pi < 0) | (fa.nix <= 0)] = np.nan
        return char


    def c_op_atl1(self):
        """Operating profits-to-lagged assets. Ball et al. (2016)"""

        # HXZ: revt - cogs - xsga + xrd.fillna(0)
        fa = self.data
        op = fa.ebitda + fa.xrd.fillna(0)
        char = op / self.shift(fa.at_)
        return char

    def c_ope_be(self):
        """Operating profits to book equity (JKP). Fama and French (2015)"""

        fa = self.data
        char = (fa.ebitda - fa.xint) / fa.be
        char[fa.be == 0] = np.nan
        return char

    def c_operprof(self):
        """Operating profits to book equity (GHZ, Org). Fama and French (2015)"""

        fa = self.data
        # char = (fa.revt - fa.cogs.fillna(0) - fa.xsga.fillna(0) - fa.xint.fillna(0)) / self.shift(fa.ceq)  # ghz
        char = (fa.revt - fa.cogs.fillna(0) - fa.xsga.fillna(0) - fa.xint.fillna(0)) / fa.ceq
        return char

    def c_ope_bel1(self):
        """Operating profits to lagged book equity. Fama and French (2015)"""

        fa = self.data
        bel1 = self.shift(fa.be)
        char = (fa.ebitda - fa.xint) / bel1
        char[bel1 == 0] = np.nan
        return char

    def c_gp_atl1(self):
        """Gross profits-to-lagged assets. Novy-Marx (2013)"""

        fa = self.data
        char = fa.gp / self.shift(fa.at_)
        return char

    def c_cop_atl1(self):
        """Cash-based operating profits to lagged assets. Ball et al. (2016)"""

        # HXZ: revt - cogs - xsga + xrd.fillna(0) - rect - invt - xpp + drc + dlrt + ap + xacc
        fa = self.data
        cop = fa.ebitda + fa.xrd.fillna(0) - fa.oacc
        char = cop / self.shift(fa.at_)
        return char

    def c_at_be(self):
        """Book leverage. Fama and French (1992)"""

        fa = self.data
        char = fa.at_ / fa.be
        char[fa.be == 0] = np.nan
        return char

    def c_aliq_at(self):
        """Asset liquidity to book assets. Ortiz-Molina and Phillips (2014)"""

        # JKP don't subtract gdwl since it's already included in intangibles. We follow JKP.
        fa = self.data
        aliq = fa.che + 0.75 * (fa.act - fa.che) + 0.5 * (fa.at_ - fa.act - fa.intan.fillna(0))
        char = aliq / self.shift(fa.at_)
        return char

    def c_aliq_mat(self):
        """Asset liquidity to market assets. Ortiz-Molina and Phillips (2014)"""

        # JKP don't subtract gdwl since it's already included in intangibles. We follow JKP.
        fa = self.data
        aliq = fa.che + 0.75 * (fa.act - fa.che) + 0.5 * (fa.at_ - fa.act - fa.intan.fillna(0))
        char = aliq / self.shift(fa.at_ - fa.be + fa.me)
        return char

    def c_tangibility(self):
        """Tangibility. Hahn and Lee (2009)"""

        fa = self.data
        return (fa.che + 0.715 * fa.rect + 0.547 * fa.invt + 0.535 * fa.ppegt) / fa.at_

    def c_eq_dur(self):
        """Equity duration. Dechow, Sloan, and Soliman (2004)"""

        fa = self.data

        T = 10
        r = 0.12

        roe_mean = 0.12
        roe_b1 = 0.57
        roe = fa.ib / np.where(self.shift(fa.be) > 1, self.shift(fa.be), np.nan)
        roe_c = roe_mean * (1 - roe_b1)

        g_mean = 0.06
        g_b1 = 0.24
        g = np.where(self.shift(fa.sale) > 1, self.pct_change(fa.sale), np.nan)
        g_c = g_mean * (1 - g_b1)

        be = fa.be
        ed_cdw = 0
        ed_cd = 0
        ed_err = np.zeros(be.shape)

        for t in range(1, T + 1):
            roe, g = roe_c + roe_b1 * roe, g_c + g_b1 * g
            be, cd = be * (1 + g), be * (roe - g)
            ed_cdw = ed_cdw + t * cd / (1 + r) ** t
            ed_cd = ed_cd + cd / (1 + r) ** t
            ed_err = np.where(be < 0, 1, ed_err)

        edc = T + 1 + 1 / r
        eq_dur = ed_cdw / fa.me + edc * (fa.me - ed_cd) / fa.me
        eq_dur = np.where((ed_err == 1) | (eq_dur <= 1), np.nan, eq_dur)
        return eq_dur

    def c_f_score(self):
        """Piotroski F-Score (JKP). Piotroski (2000)"""

        fa = self.data

        at_1 = self.shift(fa.at_, 1)

        f_roa = fa.ib / at_1
        f_croa = fa.ocf / at_1
        f_droa = self.diff(f_roa)
        f_acc = f_croa - f_roa
        f_liq = self.diff(fa.act / fa.lct)
        f_gm = self.diff(fa.gp / fa.sale)
        f_aturn = self.diff(fa.sale / at_1)
        if REPLICATE_JKP:
            f_lev = self.diff(fa.dltt / fa.at_)  # jkp use at
            f_eqis = fa.sstk  # jkp omit pstk
        else:
            f_lev = self.diff(fa.dltt / fa.avgat)  # jkp use at
            f_eqis = (fa.sstk - fa.pstk).fillna(0)  # jkp omit pstk

        char = (f_roa > 0).astype(int) + (f_croa > 0).astype(int) + (f_droa > 0).astype(int) + (f_acc > 0).astype(int) \
               + (f_lev < 0).astype(int) + (f_liq > 0).astype(int) + (f_eqis == 0).astype(int) + (f_gm > 0).astype(int) \
               + (f_aturn > 0).astype(int)
        return char

    def c_ps(self):
        """Piotroski score (GHZ, Org). Piotroski (2000)"""

        fa = self.data
        act = fa.act.fillna(fa.che + fa.rect + fa.invt)
        lct = fa.lct.fillna(fa.ap)
        char = (fa.ib > 0).astype(int) + \
               (fa.oancf > 0).astype(int) + \
               (self.diff(fa.ib / fa.at_) > 0).astype(int) + \
               (fa.oancf > fa.ib).astype(int) + \
               (self.diff(fa.dltt / fa.at_) < 0).astype(int) + \
               (self.diff(act / lct) > 0).astype(int) + \
               (self.diff(fa.sale - fa.cogs) / fa.sale > 0).astype(int) + \
               (self.diff(fa.sale / fa.at_) > 0).astype(int) + \
               (fa.scstkc == 0).astype(int)
        return char

    def c_o_score(self):
        """Ohlson O-Score. Dichev (1998)"""

        fa = self.data

        at = fa.at_.copy()
        # at[at <= 0] = np.nan
        nix = fa.nix
        nix_1 = self.shift(nix)

        o_lat = np.log(at)
        o_lev = fa.debt / at
        o_wc = (fa.act - fa.lct) / at
        o_roe = nix / at
        o_cacl = fa.lct / fa.act
        o_ffo = (fa.pi + fa.dp) / fa.lt_
        o_ffo[fa.lt_ < 0] = np.nan
        o_neg_eq = fa.lt_ > fa.at_
        o_neg_earn = (nix < 0) & (nix_1 < 0)
        o_nich = (nix - nix_1) / (nix.abs() + nix_1.abs())

        char = -1.32 - 0.407 * o_lat + 6.03 * o_lev + 1.43 * o_wc + 0.076 * o_cacl - 1.72 * o_neg_eq \
               - 2.37 * o_roe - 1.83 * o_ffo + 0.285 * o_neg_earn - 0.52 * o_nich
        return char

    def c_z_score(self):
        """Altman Z-Score. Dichev (1998)"""

        # HXZ winsorize each item at 1/99th.
        fa = self.data

        z_wc = (fa.act - fa.lct) / fa.at_
        z_re = fa.re / fa.at_
        z_eb = fa.ebitda / fa.at_  # HXZ use oiadp
        z_sa = fa.sale / fa.at_
        z_me = fa.me_fiscal / fa.lt_  # HXZ use crsp me
        z_me[fa.lt_ <= 0] = np.nan

        char = 1.2 * z_wc + 1.4 * z_re + 3.3 * z_eb + 0.6 * z_me + 1.0 * z_sa
        return char

    def c_kz_index(self):
        """Kaplan-Zingales Index. Lamont, Polk, and Saa-Requejo (2001)"""

        fa = self.data

        ppent_1 = self.shift(fa.ppent)
        ppent_1[ppent_1 <= 0] = np.nan

        kz_cf = (fa.ib + fa.dp) / ppent_1
        if REPLICATE_JKP:
            kz_q = (fa.at_ + fa.me_fiscal - fa.be) / fa.at_  # JKP omit txdb
        else:
            kz_q = (fa.at_ + fa.me_fiscal - fa.be - fa.txdb) / fa.at_  # JKP omit txdb
        # kz_q[fa.at_ <= 0] = np.nan
        kz_db = fa.debt / (fa.debt + fa.seq)
        kz_db[fa.debt + fa.seq == 0] = np.nan
        kz_dv = fa.dvt / ppent_1  # dvc + dvp
        kz_cs = fa.che / ppent_1

        char = -1.002 * kz_cf + 0.283 * kz_q + 3.139 * kz_db - 39.368 * kz_dv - 1.315 * kz_cs
        return char

    def c_intrinsic_value(self):
        """Intrinsic value-to-market. Frankel and Lee (1998)"""

        fa = self.data
        r = 0.12
        k = fa.dvt / fa.nix
        k[fa.nix <= 0] = fa.dvt / 0.06 / fa.at_
        be = self.rolling(fa.be, 2, 'mean')
        be[be <= 0] = np.nan
        Eroe = fa.nix / be
        be1 = (1 + (1 - k) * Eroe) * fa.be

        char = fa.be + (Eroe - r) / (1 + r) * fa.be + (Eroe - r) / r / (1 + r) * be1
        char[char <= 0] = np.nan
        return char

    def c_sale_emp_gr1(self):
        """Labor force efficiency. Abarbanell and Bushee (1998)"""

        fa = self.data
        char = self.pct_change(fa.sale / fa.emp)
        return char

    def c_emp_gr1(self):
        """Employee growth. Belo, Lin, and Bazdresch (2014)"""

        fa = self.data
        char = self.diff(fa.emp) / self.rolling(fa.emp, 2, 'mean')
        # char = char.fillna(0)  # if you want to set char to 0 when emp_t or emp_t-1 are missing.
        return char

    def c_earnings_variability(self):
        """Earnings smoothness. Francis et al. (2004)"""

        fa = self.data
        ni_at = fa.ib / self.shift(fa.at_)
        ocf_at = fa.ocf / self.shift(fa.at_)
        numer = self.rolling(ni_at, 5, 'std')
        denom = self.rolling(ocf_at, 5, 'std')
        char = numer / denom
        char[denom == 0] = np.nan
        return char

    def ni_ar1_ivol(self):
        """Earnings persistence and predictability"""

        fa = self.data
        n = 5  # window size

        reg = pd.DataFrame(index=fa.index)
        reg['ni_at'] = fa.ib / fa.at_
        reg['ni_at_lag1'] = self.shift(reg.ni_at)

        beta, _, ivol = self.rolling_beta(reg, n, n)
        fa['ni_ar1'] = beta[:, 1]
        fa['ni_ivol'] = ivol

    def c_ni_ar1(self):
        """Earnings persistence. Francis et al. (2004)"""

        if 'ni_ar1' not in self.data:
            self.ni_ar1_ivol()

        return self.data['ni_ar1']

    def c_ni_ivol(self):
        """Earnings predictability. Francis et al. (2004)"""

        if 'ni_ivol' not in self.data:
            self.ni_ar1_ivol()

        return self.data['ni_ivol']

####################################
    # Only in GHZ
    ####################################
    def c_cashpr(self):
        """Cash productivity. Chandrashekar and Rao (2009)"""

        fa = self.data
        return (fa.me + fa.dltt - fa.at_) / fa.che

    def c_roic(self):
        """Return on invested capital. Brown and Rowe (2007)"""

        fa = self.data
        return (fa.ebit - fa.nopi) / (fa.ceq + fa.lt_ - fa.che)

    def c_absacc(self):
        """Absolute accruals. Bandyopadhyay, Huang, and Wirjanto (2010)"""

        self.prepare(['acc'])
        fa = self.data
        return fa.acc.abs()

    def c_depr(self):
        """Depreciation to PP&E. Holthausen and Larcker (1992)"""

        fa = self.data
        return fa.dp / fa.ppent

    def c_pchdepr(self):
        """Change in depreciation to PP&E. Holthausen and Larcker (1992)"""

        fa = self.data
        char = self.pct_change(fa.dp / fa.ppent)
        return char

    def c_invest(self):
        """CAPEX and inventory. Chen and Zhang (2010)"""

        fa = self.data
        char = self.diff(fa.ppegt.fillna(fa.ppent) + fa.invt) / self.shift(fa.at_)
        return char

    def c_sin(self):
        """Sin stocks. Hong and Kacperczyk (2009)"""

        fa = self.data
        char = ((fa.sic >= '2100') & (fa.sic <= '2199')) | ((fa.sic >= '2080') & (fa.sic <= '2085')) | \
               fa.naics.isin(['7132', '71312', '713210', '71329', '713290', '72112', '721120'])
        return char.astype(int)

    def c_currat(self):
        """Current ratio. Ou and Penman (1989)"""

        fa = self.data
        act = fa.act.fillna(fa.che + fa.rect + fa.invt)
        lct = fa.lct.fillna(fa.ap)
        return act / lct

    def c_pchcurrat(self):
        """Change in current ratio. Ou and Penman (1989)"""

        self.prepare(['currat'])
        fa = self.data
        char = self.pct_change(fa.currat)
        return char

    def c_quick(self):
        """Quick ratio. Ou and Penman (1989)"""

        fa = self.data
        act = fa.act.fillna(fa.che + fa.rect + fa.invt)
        lct = fa.lct.fillna(fa.ap)
        return (act - fa.invt) / lct

    def c_pchquick(self):
        """Change in quick ratio. Ou and Penman (1989)"""

        self.prepare(['quick'])
        fa = self.data
        char = self.pct_change(fa.quick)
        return char

    def c_salecash(self):
        """Sales-to-cash. Ou and Penman (1989)"""

        fa = self.data
        return fa.sale / fa.che

    def c_salerec(self):
        """Sales-to-receivables. Ou and Penman(1989)"""

        fa = self.data
        return fa.sale / fa.rect

    def c_saleinv(self):
        """Sales-to-inventory. Ou and Penman(1989)"""

        fa = self.data
        return fa.sale / fa.invt

    def c_pchsaleinv(self):
        """Change in sales to inventory. Ou and Penman(1989)"""

        self.prepare(['saleinv'])
        fa = self.data
        char = self.pct_change(fa.saleinv)
        return char

    def c_cashdebt(self):
        """Cash flow-to-debt. Ou and Penman(1989)"""

        fa = self.data
        char = (fa.ib + fa.dp) / self.rolling(fa.lt_, 2, 'mean')
        return char

    def c_rd(self):
        """Unexpected R&D increase. Eberhart, Maxwell, and Siddique (2004)"""

        fa = self.data
        char = ((fa.xrd / fa.revt >= 0.05) & (fa.xrd / fa.at_ >= 0.05) &
                (self.pct_change(fa.xrd) >= 0.05) & (self.pct_change(fa.xrd / fa.at_) >= 0.05)).astype(int)
        return char

    def c_chpmia(self):
        """Change in profit margin. Soliman (2008)"""

        fa = self.data
        char = self.diff(fa.ib / fa.sale)
        # ghz
        # - ghz adjust for industry.
        # fa['_chpm'] = self.diff(fa.ib / fa.sale)
        # char = fa['_chpm'] - fa.groupby(['sic2', 'fyear'])['_chpm'].transform(np.mean)
        return char

    def c_chatoia(self):
        """Change in profit margin. Soliman (2008)"""

        fa = self.data
        opat = fa.rect + fa.invt + fa.aco + fa.ppent + fa.intan - fa.ap - fa.lco - fa.lo
        char = self.diff(fa.sale / self.rolling(opat, 2, 'mean'))
        # ghz
        # - ghz use avg(at) and adjust for industry.
        # fa['_chato'] = self.diff(fa.sale / fa.avgat)  # ghz
        # char = fa['_chato'] - fa.groupby(['sic2', 'fyear'])['_chato'].transform(np.mean)
        return char

    def c_bm_ia(self):
        """Industry-adjusted book-to-market. Asness, Porter, and Stevens (2000)"""

        fa = self.data
        fa['_tmp'] = fa.ceq / fa.me  # bm
        return fa['_tmp'] - fa.groupby(['sic2', 'fyear'])['_tmp'].transform(np.mean)

    def c_cfp_ia(self):
        """Industry-adjusted cash flow-to-price. Asness, Porter, and Stevens (2000)"""

        self.prepare(['cfp'])
        fa = self.data
        return fa.cfp - fa.groupby(['sic2', 'fyear']).cfp.transform(np.mean)

    def c_chempia(self):
        """Industry-adjusted change in employees. Asness, Porter, and Stevens (2000)"""

        fa = self.data
        fa['_tmp'] = self.pct_change(fa.emp).fillna(0)
        return fa['_tmp'] - fa.groupby(['sic2', 'fyear'])['_tmp'].transform(np.mean)

    def c_mve_ia(self):
        """Industry-adjusted firm size. Asness, Porter, and Stevens (2000)"""

        fa = self.data
        return fa.me - fa.groupby(['sic2', 'fyear']).me.transform(np.mean)

    ####################################
    # Only in GHZ and HXZ
    ####################################
    def c_realestate(self):
        """Real estate holdings. Tuzel (2010)"""

        fa = self.data
        fa['_rer'] = ((fa.fatb + fa.fatl) / fa.ppegt).fillna((fa.ppenb + fa.ppenls) / fa.ppent)
        gb = fa.groupby(['fyear', 'sic2'])
        char = fa['_rer'] - gb['_rer'].transform('mean')
        char[gb['_rer'].transform('count') < 5] = np.nan
        return char

    def c_securedind(self):
        """Secured debt indicator. Valta (2016)"""

        self.prepare(['secured'])
        fa = self.data
        char = (fa.secured > 0).astype(int)
        return char

    def c_secured(self):
        """Secured debt-to-total debt. Valta (2016)"""

        fa = self.data
        char = (fa.dm / (fa.dltt + fa.dlc)).fillna(0)
        return char

    def c_convind(self):
        """Convertible debt indicator. Valta (2016)"""

        fa = self.data
        fa['dc'] = fa.dcvt.fillna(fa.dcpstk - fa.pstk.fillna(0))
        fa.loc[fa.dc < ZERO, 'dc'] = np.nan
        char = ((fa.dc > 0) | (fa.cshrc > 0)).astype(int)
        return char

    def c_pchcapx_ia(self):
        """Industry-adjusted change in capital investment. Abarbanell and Bushee (1998)"""

        fa = self.data
        capx = fa.capx.fillna(self.diff(fa.ppent))
        fa['_pchcapx'] = self._chg_to_exp(capx)
        char = fa['_pchcapx'] - fa.groupby(['sic2', 'fyear'])['_pchcapx'].transform(np.mean)
        return char

    def c_tb(self):
        """Taxable income to income (Org, GHZ). Lev and Nissim (2004)"""

        # This is the original definition and requires updates on tax rates. Another implementation ('pi_nix') uses
        # pretax income and doesn't require tax rate update. GHZ industry-adjust the result, but we don't.
        fa = self.data
        tax_rate = pd.Series(0.35, index=fa.index)
        tax_rate[fa.fyear < 1993] = 0.34
        tax_rate[fa.fyear == 1987] = 0.40
        tax_rate[fa.fyear < 1987] = 0.46
        tax_rate[fa.fyear < 1979] = 0.48

        numer = (fa.txfo + fa.txfed).fillna(fa.txt - fa.txdi)
        char = numer / (tax_rate * fa.ib)
        char[(numer > 0) & (fa.ib <= 0)] = 1
        return char

    def _herf(self, char):
        # Industry concentration. Hou and Robinson (2006)

        fa = self.data
        fa['_sic3'] = fa.sic.str[:3]  # first three digit
        gb = fa.groupby(['_sic3', 'fyear'])

        fa['_tmp'] = (fa[char] / gb[char].transform(np.sum)) ** 2
        char = gb['_tmp'].transform(np.sum)

        cnt = gb['_tmp'].transform('count')

        # financial firms (6000 - 6999), railroads (4011), trucking (4210 and 4213),
        # airlines (4512), telecommunication (4812, and 4813), gas and electric utilities (4900 to 4939).
        exec_idx = ((fa.sic >= '6000') & (fa.sic <= '6999')) | \
                   (fa.sic.isin(['4011', '4210', '4213']) & (fa['datadate'] <= '1980')) | \
                   (fa.sic.isin(['4512']) & (fa['datadate'] <= '1978')) | \
                   (fa.sic.isin(['4812', '4813']) & (fa['datadate'] <= '1982')) | \
                   ((fa.sic >= '4900') & (fa.sic <= '4939')) | \
                   (cnt < 5) | (cnt / gb['_tmp'].transform('size') < 0.8)

        char.loc[exec_idx] = np.nan
        return self.rolling(char, 3, 'mean')

    def c_herf_sale(self):
        """Industry concentration (sales). Hou and Robinson (2006)"""

        return self._herf('sale')

    def c_herf_at(self):
        """Industry concentration (total assets). Hou and Robinson (2006)"""

        return self._herf('at_')

    def c_herf_be(self):
        """Industry concentration (book equity). Hou and Robinson (2006)"""

        return self._herf('be')

    ####################################
    # Diff in GHZ
    ####################################
    def c_dy(self):
        """Dividend yield (GHZ). Litzenberger and Ramaswamy (1979)"""

        fa = self.data
        return fa.dvt / fa.me

    def c_chcsho(self):
        """Net stock issues (GHZ). Pontiff and Woodgate (2008)"""

        fa = self.data
        char = self.pct_change(fa.csho * fa.ajex)
        return char

    def c_lgr(self):
        """Change in long-term debt. Richardson et al. (2005)"""

        fa = self.data
        char = self.pct_change(fa.lt_)
        return char

    # JKP definition
    # def c_ni_inc8q(self):
    #     """Number of consecutive quarters with earnings increases. Barth, Elliott, and Finn (1999)"""
    #     fq = self.data
    #     ni_inc = (self.diff(fq.ib, 1) > 0).astype(int)
    #     char = ni_inc
    #     for i in range(1, 8):
    #         idx = char == i
    #         char[idx] += ni_inc.shift(i*3)[idx]  # assume data is populated to monthly.
    #
    #     return char

################################################################################
#
# FUNDQ
#
################################################################################
class FUNDQ(Panel):
    """Class to generate firm characteristics from fundq.

    The firm characteristics generated in this class can be viewed using ``FUNDQ.show_available_functions()``:

    >>> FUNDQ().show_available_functions()

    Refer to the manual for usage.

    Args:
        alias: Characteristic column name in ``mapping.xlsx``. If None, function names (without 'c\_') are used as the
            characteristic names.
        data: DataFrame with index = date/id, sorted on id/date.
        freq: Frequency of `data`. ANNUAL, QUARTERLY, MONTHLY, or DAILY. Default to QUARTERLY.
    """

    def __init__(self, alias=None, data=None, freq=QUARTERLY):
        super().__init__(alias, data, freq, QUARTERLY)

    def load_data(self, sdate=None, edate=None):
        """Load fundq data from file.

        The loaded data is sorted on gvkey/datadate, has index = datadate/gvkey, and stored in the `data` attribute.
        """

        elapsed_time(f'Loading fundq...')
        data = WRDS.read_data('fundq')
        if sdate:
            data = data[data.datadate >= sdate]
        if edate:
            data = data[data.datadate <= edate]

        self.data = data.set_index(['datadate', 'gvkey']).sort_index(level=['gvkey', 'datadate'])

        elapsed_time('fundq loaded.')
        self.inspect_data()

    def remove_duplicates(self):
        """Drop duplicates.

        Remove duplicates in the following manner:

            1. Remove records with missing fqtr.
            2. Choose the latest record in the sense that it has the maximum fyearq and the minimum fqtr.
        """
        elapsed_time(f'Removing duplicates in fundq...')

        fq = self.data
        log(f'fundq shape before duplicate removal: {fq.shape}, duplicates: {np.sum(fq.index.duplicated())}')

        fq.dropna(subset=['fqtr'], inplace=True)  # fqtr should exist.
        fq = fq[fq.fyearq == fq.groupby(['datadate', 'gvkey']).fyearq.transform('max')]  # fyearq == max(fyearq)
        fq = fq[fq.fqtr == fq.groupby(['datadate', 'gvkey']).fqtr.transform('min')]  # fqtr == min(fqtr)

        elapsed_time(f'Duplicates removed.')
        log(f'fundq shape after duplicate removal: {fq.shape}, duplicates: {np.sum(fq.index.duplicated())}')

        self.data = fq

    def convert_currency(self):
        """Convert the currency of `data` to USD.

        This method needs to be called if

            i) the data contains non USD-denominated firms, e.g., CAD, and
            ii) CRSP's market equity is used, which is always in USD.
        """

        if (self.data.curcdq != 'USD').any():
            self.data = WRDS.convert_fund_currency_to_usd(self.data, table='fundq')

    def convert_to_monthly(self, lag=4, limit=3):
        """Populate `data` to monthly frequency.

        The index of `data` changes from datadate/gvkey to date/gvkey and datadate is kept as a column.
        The date and datadate has a gap of at least `lag` months.

        Args:
            lag: Minimum months between date and datadate: `lag` = 4 assumes funda is available 4 months after datadate.
            limit: Maximum months to forward-fill the data.
        """

        elapsed_time(f'Populating fundq data to monthly frequency...')
        # Populate data
        fq = self.data.reset_index()
        fq['date'] = fq['datadate']
        fq = fq.set_index(['date', 'gvkey'])
        fq = populate(fq, MONTHLY, limit=limit)

        # Shift data by 'lag' months.
        ids = fq.index.get_level_values('gvkey')
        fq = fq.shift(lag)
        fq = fq[ids == np.roll(ids, lag)]

        self.data = fq
        self.freq = MONTHLY
        elapsed_time('fundq data populated.')

    def _get_ytd_columns(self):
        """Get year-to-data columns."""

        return [col for col in self.data.columns if (col[-1] == 'y') and (col != 'gvkey')]

    def create_qitems_from_yitems(self):
        """Quarterize ytd items.

        Quarterize a ytd column, Xy, and fill missing Xq if Xq exists, otherwise, create a new column Xq.
        """

        self.add_row_count()
        fq = self.data
        ycolumns = self._get_ytd_columns()
        qtr_data = self.diff(fq[ycolumns])
        qtr_data[fq.fqtr == 1] = fq.loc[fq.fqtr == 1, ycolumns].values
        elapsed_time('ytd variables quarterized.')

        for col in ycolumns:
            qcol = col[:-1] + 'q'
            if qcol in fq.columns:
                fq[qcol].fillna(qtr_data[col], inplace=True)
            else:  # create new q-item.
                fq[qcol] = qtr_data[col]

        self.data = fq
        elapsed_time('Quarterly items created from ytd items.')

    def update_variables(self):
        """Preprocess variables."""

        elapsed_time(f'Updating fundq variables...')
        self.add_row_count()
        fq = self.data

        # Cleansing
        # non-negative variables with a negative value -> set to nan.
        for var in ['atq', 'saleq', 'saley', 'revtq', 'cheq']:
            fq.loc[fq[var] < 0, var] = np.nan  # should be >= 0.

        seq = fq.seqq.fillna(fq.ceqq + fq.pstkq).fillna(fq.atq - fq.ltq)
        fq['be'] = seq + fq.txditcq.fillna(0) - fq.pstkq.fillna(0)

        # tiny values -> set to 0.
        for var in fq.columns:
            if fq[var].dtype == float:
                fq[var] = np.where(is_zero(fq[var]), 0, fq[var])

        fq.prccq = fq.prccq.abs()

        # non-negative variables with a negative value -> set to nan.
        for var in ['be']:
            fq.loc[fq[var] < 0, var] = np.nan  # should be >= 0.


        elapsed_time('fundq variables updated.')

    def postprocess(self):
        """Postprocess data.

        This method deletes temporary variables (variables starting with '\_') and replaces infinite values with nan.
        This method can be overridden to add code to trim or winsorize characteristics.
        """

        elapsed_time('Postprocessing fundq...')
        fq = self.data

        # drop temporary columns that start with '_'
        del_cols = [col for col in fq if col[0] == '_']
        drop_columns(fq, del_cols)

        columns = fq.columns.intersection(self.get_available_chars())
        inspect_data(fq[columns], option=['summary', 'nans'])

        log('Replacing inf with nan...')
        fq.replace([np.inf, -np.inf], np.nan, inplace=True)

        elapsed_time('fundq postprocessed.')

    def generate_funda_vars(self):
        """Generate quarterly-updated annual data from fundq.

        The output can be merged with funda using ``FUNDA.merge_with_fundq()``.

        Returns:
            DataFrame of quarterly-updated annual data.
        """

        elapsed_time('Generating funda variables from fundq...')

        cum_columns = [  # cumulative columns
            'cogs', 'xsga', 'xint', 'dp', 'txt', 'xrd', 'spi', 'sale', 'revt',
            'xopr', 'oibdp', 'oiadp', 'ib', 'ni', 'xido', 'nopi', 'mii', 'pi', 'xi',
            'oancf', 'dv', 'sstk', 'dlcch', 'capx', 'dltr', 'txbcof', 'xidoc', 'dpc',  # ytd
            'fiao', 'ibc', 'prstkc', 'fincf',  # ytd
        ]

        fq = self.data
        column_mapping = {col: col[:-1] for col in fq.columns
                          if ((col[-1] == 'q') & (col not in ('fyearq', 'rdq')))  # q columns
                          }
        column_mapping['prccq'] = 'prcc_f'

        fa = fq.rename(columns=column_mapping)
        fa[cum_columns] = self.rolling(fa[cum_columns], 4, 'sum')

        # For the 4th fiscal quarter, use ytd data instead of the sum of quarterized values. Otherwise, data within the
        # first four quarters will be lost.
        ycolumns = [col for col in self._get_ytd_columns() if col[:-1] in cum_columns]  # columns for ytd data (un-quarterized)
        ycum_columns = [col[:-1] for col in ycolumns]
        fa.loc[fa.fqtr == 4, ycum_columns] = fa.loc[fa.fqtr == 4, ycolumns]

        elapsed_time('funda variables generated.')
        return fa

    #######################################
    # Characteristics
    #######################################
    def c_chtx(self):
        """Tax expense surprise. Thomas and Zhang (2011)"""

        fq = self.data
        numer = self.diff(fq.txtq, 4)
        denom = self.shift(fq.atq, 4)
        char = numer / denom
        char[denom == 0] = np.nan
        return char

    def c_ni_inc8q(self):
        """Number of consecutive quarters with earnings increases. Barth, Elliott, and Finn (1999)"""

        fq = self.data
        ni_inc = (self.diff(fq.ibq, 4) > 0).astype(int)
        ni_inc = self.remove_rows(ni_inc, 4)  # Need this as `nan > 0` returns False not nan.
        char = ni_inc.copy()
        for i in range(1, 8):
            idx = char == i  # True if earnings have increased so far.
            char[idx] += self.shift(ni_inc, i)[idx]

        return char

    def c_niq_be(self):
        """Return on equity (quarterly). Hou, Xue, and Zhang (2015)"""

        fq = self.data
        bel1 = self.shift(fq.be)
        char = fq.ibq / bel1
        char[bel1 == 0] = np.nan
        return char

    def c_niq_be_chg1(self):
        """Change in quarterly return on equity. Balakrishnan, Bartov, and Faurel (2010)"""

        self.prepare(['niq_be'])
        fq = self.data
        char = self.diff(fq.niq_be, 4)
        return char

    def c_niq_at(self):
        """Quarterly return on assets. Balakrishnan, Bartov, and Faurel (2010)"""

        fq = self.data
        atql1 = self.shift(fq.atq)
        char = fq.ibq / atql1
        char[atql1 == 0] = np.nan
        return char

    def c_roavol(self):
        """ROA volatility. Francis et al. (2004)"""
        self.prepare(['niq_at'])
        fq = self.data
        char = self.rolling(fq.niq_at, 16, 'std', min_n=8)
        return char

    def c_niq_at_chg1(self):
        """Change in quarterly return on assets. Balakrishnan, Bartov, and Faurel (2010)"""

        self.prepare(['niq_at'])
        fq = self.data
        char = self.diff(fq.niq_at, 4)
        return char

    # def c_saleq_gr1(self):
    #     # Quarterly sales growth
    #     fa = self.data
    #     char = self.pct_change(fa.saleq, 4)
    #     return char

    def c_saleq_su(self):
        """Revenue surprise. Jegadeesh and Livnat (2006)"""

        fq = self.data
        if REPLICATE_JKP:
            rev_diff = self.diff(fq.saleq, 4)
        else:
            rev_diff = self.diff(fq.saleq / fq.cshprq, 4)
        rev_diff_mean = self.rolling(rev_diff, 8, 'mean', min_n=6, lag=1)
        rev_diff_std = self.rolling(rev_diff, 8, 'std', min_n=6, lag=1)
        char = (rev_diff - rev_diff_mean) / rev_diff_std
        char[rev_diff_std < ZERO] = np.nan

        return char

        # HXZ
        # rev_diff = self.diff(fq.saleq / (fq.cshprq * fq.ajexq), 4)
        # char = rev_diff / self.rolling(rev_diff, 8, 'std', min_n=6)
        # return char

    def c_niq_su(self):
        """Earnings surprise. Foster, Olsen, and Shevlin (1984)"""

        fq = self.data
        if REPLICATE_JKP:
            ear_diff = self.diff(fq.ibq, 4)
        else:
            ear_diff = self.diff(fq.epspxq, 4)
        ear_diff_mean = self.rolling(ear_diff, 8, 'mean', min_n=6, lag=1)
        ear_diff_std = self.rolling(ear_diff, 8, 'std', min_n=6, lag=1)
        char = (ear_diff - ear_diff_mean) / ear_diff_std
        char[ear_diff_std < ZERO] = np.nan

        return char

        # HXZ
        # ear_diff = self.diff(fq.epspxq / fq.ajexq, 4)
        # char = ear_diff / self.rolling(ear_diff, 8, 'std', min_n=6)
        # return char

    def c_ocfq_saleq_std(self):
        """Cash flow volatility. Huang (2009)"""

        fq = self.data
        if REPLICATE_JKP:
            oancfq = fq.oancfq.fillna(fq.ibq + fq.dpq)
        else:
            oancfq = fq.oancfq.fillna(fq.ibq + fq.dpq - self.diff(fq.wcapq.fillna(0)))  # JKP don't subtract d(wcapq)
        char = self.rolling(oancfq / fq.saleq, 16, 'std', min_n=8)
        return char

    # def c_cash(self):  # already defined in funda as cash_at
    #     # Cash holdings. Palazzo (2012)
    #     fq = self.data
    #     return fq.cheq / fq.atq

    ####################################
    # Only in GHZ
    ####################################
    def c_rsup(self):
        """Revenue surprise (Karma). Kama (2009)"""

        fq = self.data
        char = self.diff(fq.saleq, 4) / (fq.prccq * fq.cshoq)
        return char

    def c_stdacc(self):
        """Accrual volatility. Bandyopadhyay, Huang, and Wirjanto (2010)"""

        fq = self.data
        saleq = fq.saleq.copy()
        saleq.loc[saleq <= 0] = 0.01
        sacc = self.diff(fq.actq - fq.cheq - fq.lctq + fq.dlcq) / saleq
        char = self.rolling(sacc, 16, 'std')
        return char


################################################################################
#
# CRSPM
#
################################################################################
class CRSPM(Panel):
    """Class to generate firm characteristics from crspm.

    The firm characteristics generated in this class can be viewed using ``CRSPM.show_available_functions()``:

    >>> CRSPM().show_available_functions()

    Refer to the manual for usage.

    Args:
        alias: Characteristic column name in ``mapping.xlsx``. If None, function names (without 'c\_') are used as the
            characteristic names.
        data: DataFrame with index = date/id, sorted on id/date.
        freq: Frequency of `data`. ANNUAL, QUARTERLY, MONTHLY, or DAILY. Default to MONTHLY.
    """

    def __init__(self, alias=None, data=None, freq=MONTHLY):
        super().__init__(alias, data, freq, MONTHLY)

    def load_data(self, sdate=None, edate=None):
        """Load crspm data from file.

        The loaded data is sorted on permno/date, has index = date/permno, and stored in the `data` attribute.

        NOTE:
            In CRSP monthly tables, date is the last business day of the month, whereas datadate in Compustat is the
            end-of-month date. To make the two dates consistent, crspm.date is shifted to the end of the month.
        """

        elapsed_time(f'Loading crspm...')
        data = WRDS.read_data('crspm')
        if sdate:
            data = data[data.date >= sdate]
        if edate:
            data = data[data.date <= edate]

        data['date'] = to_month_end(data['date'])  # Push date to month end ignoring holidays. compustat datadate is
                                                   # always month end, while crsp date is business day month end.

        data.sort_values(['permno', 'date'], inplace=True)
        data.set_index(['date', 'permno'], inplace=True)
        self.data = data

        elapsed_time(f'crspm loaded.')
        self.inspect_data()

    def filter_data(self):
        """Filter data.

        Currently, we filter data only on `shrcd`.
        """

        elapsed_time('Filtering crspm...')
        log(f'crspm shape before filtering: {self.data.shape}')

        # Ordinary Common Shares: shrcd in (10, 11, 12).
        self.filter(('shrcd', 'in', [10, 11, 12]))

        # Exchange filtering: 1 (NYSE), 2 (ASE), 3 (NASDAQ)
        # We prefer to not filter on exchcd as it can change in the month when a stock is delisted.
        # self.filter(('exchcd', 'in', [1, 2, 3]))

        log(f'crspm shape after filtering: {self.data.shape}')
        elapsed_time('crspm data filtered.')

    def update_variables(self):
        """Update variables."""

        elapsed_time('Updating crspm variables...')

        self.add_row_count()
        cm = self.data.reset_index()

        cm.prc = cm.prc.abs()
        cm.shrout /= 1000  # to millions
        cm.vol *= 100  # to number of shares. unit of vol: daily data -> number of shares; monthly data -> 100 shares.

        # Adjust trading volume following Gao and Ritter (2010)
        date = cm['date']
        cm.loc[(cm.exchcd == 3) & (date <= '2001-01-31'), 'vol'] /= 2
        cm.loc[(cm.exchcd == 3) & (date > '2001-01-31') & (date <= '2001-12-31'), 'vol'] /= 1.8
        cm.loc[(cm.exchcd == 3) & (date > '2001-12-31') & (date <= '2003-12-31'), 'vol'] /= 1.6

        # Market equity
        cm['me'] = cm.prc * cm.shrout
        # cm['me_nyse'] = cm.loc[cm.exchcd == 1, 'me']  # NYSE: exchcd = 1
        cm['me_company'] = cm.groupby(['date', 'permco'])['me'].transform(sum)  # firm-level market equity
        cm.loc[cm.me.isna(), 'me_company'] = np.nan

        # Risk-free rate and excess return
        rf = WRDS.get_risk_free_rate(month_end=True)
        cm = cm.merge(rf, on='date', how='left')
        cm.set_index(['date', 'permno'], inplace=True)
        cm['exret'] = winsorize(cm.ret - cm.rf, (1e-3, 1e-3))  # winsorize at 0.1%, 99.9%

        self.data = cm
        elapsed_time('crspm variables updated.')

    def merge_with_factors(self, factors=None):
        """Merge crspm with factors.

        The `factors` should contain Fama-French 3 factors with column names as defined in config.factor_names.

        Args:
            factors: DataFrame of factors with index = date. If None, it will be read from config.monthly_factors_fname.
        """

        elapsed_time('Merging factors with crspm...')
        if factors is None:
            factors = read_from_file(config.factors_monthly_fname)

        # Rename factor names as the keys of config.factor_names.
        factors = factors.rename(columns={v: k for k, v in config.factor_names.items()})
        if 'rf' in factors:  # drop rf as it is already in crspm.
            del factors['rf']

        factors.index = to_month_end(factors.index)
        self.merge(factors, on=['date'], how='left')
        elapsed_time('Factors merged with crspm.')

    def postprocess(self):
        """Postprocess data.

        This method deletes temporary variables (variables starting with '\_') and replaces infinite values with nan.
        This method can be overridden to add code to trim or winsorize characteristics.
        """

        elapsed_time('Postprocessing crspm...')
        cm = self.data

        # drop temporary columns that start with '_'
        del_cols = [col for col in cm if col[0] == '_']
        drop_columns(cm, del_cols)

        columns = cm.columns.intersection(self.get_available_chars())
        inspect_data(cm[columns], option=['summary', 'nans'])

        log('Replacing inf with nan...')
        cm.replace([np.inf, -np.inf], np.nan, inplace=True)

        elapsed_time('crspm postprocessed.')

    #######################################
    # Characteristics
    #######################################
    def c_market_equity(self):
        """Market equity. Banz (1981)"""
        cm = self.data
        return cm.me

    def c_div12m_me(self):
        """Dividend yield (JKP). Litzenberger and Ramaswamy (1979)"""

        cm = self.data
        div_tot = (cm.ret - cm.retx) * self.shift(cm.prc) * cm.cfacshr / self.shift(cm.cfacshr)
        char = self.rolling(div_tot * cm.shrout, 12, 'sum') / cm.me
        char[cm.me == 0] = np.nan
        return char

    def c_chcsho_12m(self):
        """Net stock issues (JKP). Pontiff and Woodgate (2008)"""

        cm = self.data
        char = self.pct_change(cm.shrout * cm.cfacshr, 12)
        return char

    def c_eqnpo_60m(self):
        """Composite equity issuance (Org). Daniel and Titman (2006)"""

        cm = self.data
        char = np.log(cm.me / self.shift(cm.me, 60)) - np.log(self.cumret(cm.ret, 60) + 1)
        return char

    def c_eqnpo_12m(self):
        """Composite equity issuance (JKP, 12 months). Daniel and Titman (2006)"""

        # This is equal to (-1) * JKP's eqnpo_12m.
        cm = self.data
        char = np.log(cm.me / self.shift(cm.me, 12)) - np.log(self.cumret(cm.ret, 12) + 1)
        if REPLICATE_JKP:
            return -char
        else:
            return char

    def _mom_i_j(self, i, j):
        char = self.cumret(self.data.ret, i, j)
        return char

    def c_ret_1_0(self):
        """Short-term reversal. Jegadeesh (1990)"""

        return self._mom_i_j(1, 0)

    def c_ret_3_1(self):
        """Momentum (3 months). Jegadeesh and Titman (1993)"""

        return self._mom_i_j(3, 1)

    def c_ret_6_1(self):
        """Momentum (6 months). Jegadeesh and Titman (1993)"""

        return self._mom_i_j(6, 1)

    def c_ret_9_1(self):
        """Momentum (9 months). Jegadeesh and Titman (1993)"""

        return self._mom_i_j(9, 1)

    def c_ret_12_1(self):
        """Momentum (12 months). Jegadeesh and Titman (1993)"""

        return self._mom_i_j(12, 1)

    def c_ret_12_6(self):
        """Intermediate momentum (7-12). Novy-Marx (2012)"""

        return self._mom_i_j(12, 6)

    def c_ret_36_12(self):
        """Long-term reversal (12-36). De Bondt and Thaler (1985)"""

        return self._mom_i_j(36, 12)

    def c_ret_60_12(self):
        """Long-term reversal (12-60). De Bondt and Thaler (1985)"""

        return self._mom_i_j(60, 12)

    def c_indmom(self):
        """Industry momentum. Moskowitz and Grinblatt (1999)"""

        cm = self.data
        cm = cm[['ret', 'me', 'hsiccd']].copy()
        cm['sic2'] = cm['hsiccd'] // 100  # First two-digits. The type of siccd is float in crsp.
        cm['ret6'] = self.cumret(cm.ret, 6)
        indmom = weighted_mean(cm, ['ret6'], 'me', ['date', 'sic2'])
        indmom.columns = ['indmom']
        cm = cm.reset_index().merge(indmom, on=['date', 'sic2'], how='left')

        return cm.set_index(['date', 'permno'])['indmom']

    def _seasonality(self, ret, start, end, an=True):
        dur = end - start + 1
        offset = 12 * start - 1

        ret_sum = 0
        for i in range(dur):
            ret_sum += self.shift(ret, offset + 12 * i)

        if an:
            char = ret_sum / dur
        else:
            char = (self.rolling(ret, 12 * dur, 'sum', lag=offset - 11) - ret_sum) / (12 - 1) / dur
        return char

    def c_seas_1_1an(self):
        """Year 1-lagged return, annual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 1, 1, True)

    def c_seas_2_5an(self):
        """Years 2-5 lagged returns, annual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 2, 5, True)

    def c_seas_6_10an(self):
        """Years 6-10 lagged returns, annual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 6, 10, True)

    def c_seas_11_15an(self):
        """Years 11-15 lagged returns, annual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 11, 15, True)

    def c_seas_16_20an(self):
        """Years 16-20 lagged returns, annual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 16, 20, True)

    def c_seas_1_1na(self):
        """Year 1-lagged return, nonannual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 1, 1, False)

    def c_seas_2_5na(self):
        """Years 2-5 lagged returns, nonannual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 2, 5, False)

    def c_seas_6_10na(self):
        """Years 6-10 lagged returns, nonannual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 6, 10, False)

    def c_seas_11_15na(self):
        """Years 11-15 lagged returns, nonannual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 11, 15, False)

    def c_seas_16_20na(self):
        """Years 16-20 lagged returns, nonannual. Heston and Sadka (2008)"""

        cm = self.data
        return self._seasonality(cm.ret, 16, 20, False)

    def c_beta_60m(self):
        """Market beta (Org, JKP). Fama and MacBeth (1973)"""

        cm = self.data
        # return rolling_simple_beta(cm, 'exret', 'mktrf', 60, 36)
        # cm = cm[['exret', 'mktrf']].copy()
        # cm['exret'] = winsorize(cm['exret'], (0.001, 0.001))
        if REPLICATE_JKP:
            cm.loc[cm.ret == 0, 'exret'] = np.nan  # JKP omit zero returns.

        beta, r2, idio = self.rolling_beta(cm[['exret', 'mktrf']], 60, 36)
        return beta[:, 1]

    def c_prc(self):
        """Share price. Miller and Scholes (1982)"""

        return self.data['prc']

    @staticmethod
    # @multiprocess
    def resff3_mom(cm, window=36, minobs=24):
        ngroups = cm.shape[0]
        gsize = cm.groupby(['permno']).size().values
        data = cm.values  # 'exret', 'mktrf', 'hml', 'smb_ff'
        isnan = np.isnan(data).any(axis=1)

        @njit(error_model='numpy')
        def fcn(data, gsize, ngroups):
            beta = np.full((ngroups, 4), np.nan)
            resff3_6_1 = np.full((ngroups, 1), np.nan)
            resff3_12_1 = np.full((ngroups, 1), np.nan)

            idx0 = 0
            for g in range(gsize.shape[0]):
                obs = gsize[g]
                if obs < minobs:
                    idx0 += obs
                    continue

                subdata = data[idx0:idx0 + obs]
                isnan_ = isnan[idx0:idx0 + obs]

                for i in range(minobs - 1, obs):
                    if i < window:
                        subvals = subdata[0:i + 1]
                        subvals = subvals[~isnan_[0:i + 1]]
                    else:
                        subvals = subdata[i - window + 1:i + 1]
                        subvals = subvals[~isnan_[i - window + 1:i + 1]]

                    if subvals.shape[0] < minobs:
                        continue

                    endo = subvals[:, 0].copy()
                    exog = add_constant(subvals[:, 1:])

                    beta[idx0 + i, :] = (np.linalg.solve(exog.T @ exog, exog.T @ endo)).reshape(4)
                    res = endo - (beta[idx0 + i, :] * exog).sum(1)
                    resff3_6_1[idx0 + i] = res[-6:-1].mean() / std(res[-6:-1])
                    resff3_12_1[idx0 + i] = res[-12:-1].mean() / std(res[-12:-1])

                idx0 += obs
            return resff3_6_1, resff3_12_1

        return fcn(data, gsize, ngroups)

    def c_resff3_6_1(self):
        """6 month residual momentum. Blitz, Huij, and Martens (2011)"""

        if 'resff3_6_1' not in self.data:
            columns = ['exret', 'mktrf', 'hml', 'smb_ff']
            # self.data[['resff3_6_1', 'resff3_12_1']] = self.resff3_mom(self.data[columns], 36, 24)
            retval = self.resff3_mom(self.data[columns], 36, 24)
            self.data['resff3_6_1'] = retval[0]
            self.data['resff3_12_1'] = retval[1]

        return self.data['resff3_6_1']

    def c_resff3_12_1(self, mp=True):
        """12 month residual momentum. Blitz, Huij, and Martens (2011)"""

        if 'resff3_12_1' not in self.data:
            columns = ['exret', 'mktrf', 'hml', 'smb_ff']
            # self.data[['resff3_6_1', 'resff3_12_1']] = self.resff3_mom(self.data[columns], 36, 24)
            retval = self.resff3_mom(self.data[columns], 36, 24)
            self.data['resff3_6_1'] = retval[0]
            self.data['resff3_12_1'] = retval[1]

        return self.data['resff3_12_1']

    ####################################
    # Only in GHZ
    ####################################
    def c_chmom(self):
        """Change in 6-month momentum. Gettleman and Marks (2006)"""

        cm = self.data
        char = self.diff(self.cumret(cm.ret, 6), 6)
        return char

    def c_divi(self):
        """Dividend initiation. Michaely, Thaler, and Womack (1995)"""

        cm = self.data
        div = cm.cash_div.fillna(0)
        cum_div = self.shift(div).groupby('permno').cumsum()  # sum of previous dividends
        char = (cum_div == 0) & (div > 0) & (cm.rcount >= 24)
        return char.astype(int)

    def c_divo(self):
        """Dividend omission. Michaely, Thaler, and Womack (1995)"""

        cm = self.data
        div = cm.cash_div.fillna(0)
        div3 = self.shift(div, 3)  # div 3 months before
        div6 = self.shift(div, 6)
        div9 = self.shift(div, 9)
        div12 = self.shift(div, 12)
        div15 = self.shift(div, 15)
        div18 = self.shift(div, 18)
        div24 = self.shift(div, 24)  # div 24 months before
        char1 = (div == 0) & (div3 > 0) & (div6 > 0) & (div9 > 0) & (div12 > 0) & (div15 > 0) & (div18 > 0) & (
                    cm.rcount > 18)  # omission of quarterly dividend
        char2 = (div == 0) & (div9 == 0) & (div6 > 0) & (div12 > 0) & (div18 > 0) & (
                    cm.rcount > 18)  # omission of semi-annual dividend. assume semiannual if div9 = 0
        char3 = (div == 0) & (div18 == 0) & (div12 > 0) & (div24 > 0) & (
                    cm.rcount > 24)  # omission of annual dividend. assume annual if div18 = 0
        return (char1 | char2 | char3).astype(int)

    def c_dolvol(self):
        """Dollar trading volume (Org, GHZ). Brennan, Chordia, and Subrahmanyam (1998)"""

        # GHZ implementation (Original definition)
        cm = self.data
        char = self.shift(np.log(cm.vol * cm.prc))
        return char

    def c_turn(self):
        """Share turnover (Org, GHZ). Datar, Naik, and Radcliffe (1998)"""

        cm = self.data
        char = self.rolling(cm.vol, 3, 'mean') / cm.shrout
        return char

    def c_ipo(self):
        """Initial public offerings. Loughran and Ritter (1995)"""

        cm = self.data
        return (cm.rcount < 12).astype(int)


################################################################################
#
# CRSPD
#
################################################################################
class CRSPD(Panel):
    """Class to generate firm characteristics from crspd.

    CRSPD generates firm characteristics monthly and store them in the `chars` attribute instead of the `data`
    attribute, which is a daily data.

    The firm characteristics generated in this class can be viewed using ``CRSPD.show_available_functions()``:

    >>> CRSPD().show_available_functions()

    Refer to the manual for usage.

    Args:
        alias: Characteristic column name in ``mapping.xlsx``. If None, function names (without 'c\_') are used as the
            characteristic names.
        data: DataFrame with index = date/id, sorted on id/date.
        freq: Frequency of `data`. ANNUAL, QUARTERLY, MONTHLY, or DAILY. Default to DAILY.

    Attributes:
        chars: DataFrame to store firm characteristics. Index = date/permno.
    """

    def __init__(self, alias=None, data=None, freq=DAILY):
        super().__init__(alias, data, freq, DAILY)
        self.chars = None  # characteristics are generated monthly and stored here, not in self.data.

    def create_chars(self, char_list=None):
        """Generate firm characteristics.

        This method overrides ``Panel.create_chars()`` to store characteristics in `chars` instead of `data`.
        """
        char_list = char_list or self.get_available_chars()

        for char in char_list:
            # try:
            fcn = self.char_map[char]
            self.chars[fcn] = eval(f'self.c_{fcn}()')  # self.data -> self.chars
            elapsed_time(f"[{char}] created.")
            # except Exception as e:
            #     log(f'Error occured creating {char}: {e}')

    def prepare(self, char_fcns):
        """Prepare "ingredient" characteristics that are required to generate a characteristic.

        This method overrides ``Panel.prepare()`` to store characteristics in `chars` instead of `data`.
        """

        for fcn in char_fcns:
            if fcn not in self.chars:
                self.create_chars([self.reverse_map[fcn]])

    def load_data(self, sdate=None, edate=None):
        """Load crspd data from file.

        The loaded data is sorted on permno/date, has index = date/permno, and stored in the `data` attribute.
        """

        elapsed_time(f'Loading crspd...')
        data = WRDS.read_data('crspd')
        if sdate:
            data = data[data.date >= sdate]
        if edate:
            data = data[data.date <= edate]

        data.sort_values(['permno', 'date'], inplace=True)
        data.set_index(['date', 'permno'], inplace=True)
        for col in data:
            if data[col].dtype == 'float32':
                data[col] = data[col].astype('float')

        self.data = data

        elapsed_time(f'crspd loaded.')
        self.inspect_data()

    def save(self, fname=None, fdir=None, other_columns=None):
        """Save this object.

        This method overrides ``Panel.save()`` to save `chars` instead of `data`.
        """

        data = self.data
        self.data = self.chars
        super().save(fname, fdir)
        self.data = data

    def load(self, fname=None, fdir=None):
        """Load data and parameters.

        This method overrides ``Panel.load()`` to load `chars` instead of `data`.
        """

        super().load(fname, fdir)
        self.chars = self.data

    def filter_data(self):
        """Filter data.

        Currently, we filter data only on `shrcd`.
        """

        elapsed_time('Filtering crspd...')
        log(f'crspd shape before filtering: {self.data.shape}')

        # Ordinary Common Shares: shrcd in (10, 11, 12).
        self.filter(('shrcd', 'in', [10, 11, 12]))

        # Exchange filtering: 1 (NYSE), 2 (ASE), 3 (NASDAQ)
        # We prefer to not filter on exchcd as it can change in the month when a stock is delisted.
        # self.filter(('exchcd', 'in', [1, 2, 3]))

        log(f'crspd shape after filtering: {self.data.shape}')
        elapsed_time('crspd data filtered.')

    def update_variables(self):
        """Update variables."""

        elapsed_time('Updating crspd variables...')
        self.add_row_count()
        cd = self.data.reset_index().copy()
        del self.data

        cd.loc[(cd.askhi <= 0) | (cd.prc <= 0) | (cd.vol == 0), 'askhi'] = np.nan
        cd.loc[(cd.bidlo <= 0) | (cd.prc <= 0) | (cd.vol == 0), 'bidlo'] = np.nan

        cd['prc'] = cd.prc.abs()

        cd['prc_adj'] = cd.prc / cd.cfacpr
        if REPLICATE_JKP:
            cd.loc[cd.cfacpr == 0, 'cfacshr'] = np.nan  # JKP use cfacshr instead of cfacpr
            cd['prc_adj'] = cd.prc / cd.cfacshr
            cd['bidlo_adj'] = cd.bidlo / cd.cfacshr
            cd['askhi_adj'] = cd.askhi / cd.cfacshr
        else:
            cd.loc[cd.cfacpr == 0, 'cfacpr'] = np.nan
            cd['prc_adj'] = cd.prc / cd.cfacpr
            cd['bidlo_adj'] = cd.bidlo / cd.cfacpr
            cd['askhi_adj'] = cd.askhi / cd.cfacpr
        cd.shrout /= 1000  # to millions

        # Adjust trading volume following Gao and Ritter (2010)
        date = cd['date']
        cd.loc[(cd.exchcd == 3) & (date <= '2001-01-31'), 'vol'] /= 2
        cd.loc[(cd.exchcd == 3) & (date > '2001-01-31') & (date <= '2001-12-31'), 'vol'] /= 1.8
        cd.loc[(cd.exchcd == 3) & (date > '2001-12-31') & (date <= '2003-12-31'), 'vol'] /= 1.6

        cd['me'] = cd.prc * cd.shrout
        cd['me_company'] = cd.groupby(['date', 'permco'])['me'].transform(sum)  # firm-level market equity
        cd['dvol'] = cd.prc * cd.vol

        cd['mon'] = ((date.dt.year - 1900) * 12 + date.dt.month).astype('int16')
        cd['ym'] = to_month_end(date)

        # Risk-free rate and excess return
        # rf = WRDS.get_risk_free_rate().astype('float32') / 21
        rf = WRDS.get_risk_free_rate(month_end=True) / 21
        cd = cd.merge(rf, left_on='ym', right_on='date', how='left')
        cd.set_index(['date', 'permno'], inplace=True)

        if REPLICATE_JKP:
            cd['exret'] = cd.ret - cd.rf  # JKP winsorise excess return only in crspm. We do so also in crspd.
        else:
            cd['exret'] = winsorize(cd.ret - cd.rf, (1e-3, 1e-3))  # winsorize at 0.1%, 99.9% (slower)
        cd['turnover_d'] = cd.vol / (cd.shrout * 1e6)
        # cd['turnover_d'] = (cd.vol / (cd.shrout * 1e6)).astype('float32')

        # If number of zero returns >= 10 in a month, all returns of the month are set to nan.
        cd['zero_ret'] = cd['ret'] == 0
        cd['zero_ret_cnt'] = cd.groupby(['ym', 'permno']).zero_ret.transform('sum')
        cd.loc[cd.zero_ret_cnt >= 10, 'exret'] = np.nan
        cd.drop(columns=['zero_ret', 'zero_ret_cnt'], inplace=True)

        # Initialize chars.
        self.chars = pd.DataFrame(index=cd.groupby(['permno', 'ym']).last().index).swaplevel()
        self.chars.index.names = ('date', 'permno')
        self.chars['rcount'] = self.chars.groupby('permno').cumcount()

        self.data = cd
        elapsed_time('crspd variables updated.')

    def merge_with_factors(self, factors=None):
        """Merge crspd with factors.

        The `factors` should contain Fama-French 3 factors and Hou-Xue-Zhang 4 factors with column names as defined
        in config.factor_names.

        Args:
            factors: DataFrame of factors with index = date. If None, it will be read from config.daily_factors_fname.
        """

        elapsed_time('Merging factors with crspd...')
        if factors is None:
            factors = read_from_file(config.factors_daily_fname)

        # for col in factors:
        #     if factors[col].dtype == 'float':
        #         factors[col] = factors[col].astype('float32')

        # Rename factor names as the keys of config.factor_names.
        factors = factors.rename(columns={v: k for k, v in config.factor_names.items()})
        if 'rf' in factors:  # drop rf as it is already in crspm.
            del factors['rf']

        self.merge(factors, on=['date'], how='left')
        elapsed_time('Factors merged with crspd.')

    def postprocess(self):
        """Postprocess data.

        This method deletes temporary variables (variables starting with '\_') and replaces infinite values with nan.
        This method can be overridden to add code to trim or winsorize characteristics.
        """

        elapsed_time('Postprocessing crspd...')
        cd = self.chars

        # drop temporary columns that start with '_'
        del_cols = [col for col in cd if col[0] == '_']
        drop_columns(cd, del_cols)

        columns = cd.columns.intersection(self.get_available_chars())
        inspect_data(cd[columns], option=['summary', 'nans'])

        log('Replacing inf with nan...')
        cd.replace([np.inf, -np.inf], np.nan, inplace=True)

        elapsed_time('crspd postprocessed.')

    @staticmethod
    def rolling_apply_m(cd, data_columns, fcn, n_retval):
        """Apply fcn in every month. This function runs a loop over permno-month and can be used when the calculation
        inside fcn requires only the data within the month.

        Args:
            cd: crspd.data
            data_columns: columns of cd that are used as input to fcn.
            fcn(data, isnan): function to apply. data is cd[data_columns] of a permno/mon pair, and isnan is a non
            indicator: isnan[i] = True if any column of data[i, :] has nan. fcn should return a tuple of size n_retval.
            n_retval: number of returns from fcn.

        Returns:
            Concatenated ndarray of the returns from fcn. Size = permno/mon x n_retval.
        """

        gsize = cd.groupby(['permno', 'mon']).size().values
        data = cd[data_columns].values
        isnan = np.isnan(data).any(axis=1)

        @njit
        def rolling_apply_m_(data, gsize):
            ngroups = gsize.shape[0]
            retval = np.full((ngroups, n_retval), np.nan)

            idx0 = 0
            for i in range(ngroups):
                data_ = data[idx0:idx0 + gsize[i]]
                isnan_ = isnan[idx0:idx0 + gsize[i]]
                idx0 += gsize[i]

                retval[i, :] = fcn(data_, isnan_)

            return retval

        return rolling_apply_m_(data, gsize)

    #######################################
    # Characteristics
    #######################################
    def c_retvol(self):
        """Return volatility. Ang et al. (2006)"""

        cd = self.data
        return cd.groupby(['permno', 'ym']).ret.std().swaplevel()

    def c_rmax1_21d(self):
        """Maximum daily return. Bali, Cakici, and Whitelaw (2011)"""

        cd = self.data
        return cd.groupby(['permno', 'ym']).ret.max().swaplevel()

    def c_rmax5_21d(self):
        """Highest 5 days of return. Bali, Brown, and Tang (2017)"""

        if 'rmax5_21d' in self.chars:
            return self.chars['rmax5_21d']
        cd = self.data
        vars = cd[['ret', 'ym']].reset_index()
        vars['retrank'] = vars.groupby(['permno', 'ym']).ret.rank(ascending=False)
        vars['high'] = np.where(vars.retrank <= 5, vars.ret, np.nan)
        return vars.groupby(['permno', 'ym']).high.mean().swaplevel()

    def c_rskew_21d(self):
        """Return skewness. Bali, Engle, and Murray (2016)"""

        cd = self.data

        @njit
        def fcn(data, isnan):
            data = data[~isnan]
            if data.shape[0] >= 3:
                return skew(data)
            else:
                return np.nan

        return self.rolling_apply_m(cd, ['ret'], fcn, 1)

    def icapm_21d(self):
        """ivol, skew, coskew from CAPM"""
        if 'ivol_capm_21d' in self.chars:
            return

        reg_vars = ['exret', 'mktrf']  # y, X
        ocolumns = ['ivol_capm_21d', 'iskew_capm_21d', 'coskew_capm_21d']

        @njit(error_model='numpy')
        def fcn(data, isnan):
            data = data[~isnan]
            if data.shape[0] < 15:
                return np.nan, np.nan, np.nan

            endo = data[:, 0]
            exog = data[:, 1]

            cov = np.cov(endo, exog)
            beta = cov[0][1] / cov[1][1]
            c = endo.mean() - beta * exog.mean()
            pred = c + beta * exog
            res = endo - pred
            ivol = std(res)
            iskew = skew(res)

            exog_dm = exog - exog.mean()
            coskew = (res * exog_dm ** 2).mean() / ((res ** 2).mean() ** 0.5 * (exog_dm ** 2).mean())

            return ivol, iskew, coskew

        self.chars[ocolumns] = self.rolling_apply_m(self.data, reg_vars, fcn, 3)

    def c_ivol_capm_21d(self):
        """Idiosyncratic volatility (CAPM). Ang et al. (2006)"""

        self.icapm_21d()
        return self.chars['ivol_capm_21d']

    def c_iskew_capm_21d(self):
        """Idiosyncratic skewness (CAPM). Bali, Engle, and Murray (2016)"""

        self.icapm_21d()
        return self.chars['iskew_capm_21d']

    def c_coskew_21d(self):
        """Coskewness. Harvey and Siddique (2000)"""

        self.icapm_21d()
        return self.chars['coskew_capm_21d']

    @staticmethod
    def multifactor_ivol_skew(cd, reg_vars):
        """ivol and skew from a multi-factor model."""

        @njit
        def fcn(data, isnan):
            data = data[~isnan]
            if data.shape[0] < 15:
                return np.nan, np.nan

            endo = data[:, 0].copy()
            exog = add_constant(data[:, 1:])

            beta = (np.linalg.solve(exog.T @ exog, exog.T @ endo)).reshape(data.shape[1])
            pred = (beta * exog).sum(1)
            res = endo - pred

            if REPLICATE_JKP:
                return std(res) * np.sqrt((data.shape[0] - 1)/(data.shape[0] - data.shape[1] + 1)), skew(res)
            else:
                return std(res), skew(res)

        return CRSPD.rolling_apply_m(cd, reg_vars, fcn, 2)

    def iff3_21d(self):
        if 'ivol_ff3_21d' in self.chars:
            return
        reg_vars = ['exret', 'mktrf', 'hml', 'smb_ff']  # y, X
        ocolumns = ['ivol_ff3_21d', 'iskew_ff3_21d']
        self.chars[ocolumns] = self.multifactor_ivol_skew(self.data, reg_vars)

    def c_ivol_ff3_21d(self):
        """Idiosyncratic volatility (FF3). Ang et al. (2006)"""

        self.iff3_21d()
        return self.chars['ivol_ff3_21d']

    def c_iskew_ff3_21d(self):
        """Idiosyncratic skewness (FF3). Bali, Engle, and Murray (2016)"""

        self.iff3_21d()
        return self.chars['iskew_ff3_21d']

    def ihxz4_21d(self):
        if 'ivol_hxz4_21d' in self.chars:
            return
        reg_vars = ['exret', 'mktrf', 'smb_hxz', 'inv', 'roe']  # y, X
        ocolumns = ['ivol_hxz4_21d', 'iskew_hxz4_21d']
        self.chars[ocolumns] = self.multifactor_ivol_skew(self.data, reg_vars)

    def c_ivol_hxz4_21d(self):
        """Idiosyncratic volatility (HXZ). Ang et al. (2006)"""

        self.ihxz4_21d()
        return self.chars['ivol_hxz4_21d']

    def c_iskew_hxz4_21d(self):
        """Idiosyncratic skewness (HXZ). Bali, Engle, and Murray (2016)"""

        self.ihxz4_21d()
        return self.chars['iskew_hxz4_21d']

    def c_beta_dimson_21d(self):
        """Dimson Beta. Dimson (1979)"""

        cd = self.data
        cd['mktrf_lg1'] = self.shift(cd.mktrf)
        cd['mktrf_ld1'] = cd.mktrf.shift(-1)
        reg_vars = ['exret', 'mktrf', 'mktrf_lg1', 'mktrf_ld1']

        @njit
        def fcn(data, isnan):
            data = data[:-1]  # remove the last row that contains a future market return (mktrf_ld1)
            data = data[~isnan[:-1]]
            if data.shape[0] < 15:
                return np.nan

            endo = data[:, 0].copy()
            exog = add_constant(data[:, 1:])

            beta = (np.linalg.solve(exog.T @ exog, exog.T @ endo)).reshape(4)
            dimson = beta[1:].sum()

            return dimson

        char = self.rolling_apply_m(cd, reg_vars, fcn, 1)
        cd.drop(columns=['mktrf_lg1', 'mktrf_ld1'], inplace=True)
        return char

    def c_zero_trades_21d(self):
        """Zero-trading days (1 month). Liu (2006)"""

        cd = self.data
        cd['_tmp'] = cd.vol == 0
        gb = cd.groupby(['permno', 'ym'])
        cnt = gb.ym.count()
        char = (gb._tmp.sum() + 1 / gb.turnover_d.sum() / 480000) * 21 / cnt
        char[cnt == 0] = np.nan
        return char.swaplevel()

    @staticmethod
    @njit(error_model='numpy')
    def adjust_bidaskhl(prc, bidlo, askhi, rcount):
        """Adjust bidlo, askhi."""
        bidlo = bidlo.copy()
        askhi = askhi.copy()

        bidlo0 = np.nan
        askhi0 = np.nan
        for i in range(prc.shape[0]):
            if rcount[i] == 0:
                bidlo0 = np.nan
                askhi0 = np.nan

            if 0 < bidlo[i] < askhi[i]:
                bidlo0 = bidlo[i]
                askhi0 = askhi[i]
            elif bidlo0 <= prc[i] <= askhi0:
                bidlo[i] = bidlo0
                askhi[i] = askhi0
            elif prc[i] < bidlo0:
                bidlo[i] = prc[i]
                askhi[i] = askhi0 - bidlo0 + prc[i]
            elif prc[i] > askhi0:
                bidlo[i] = bidlo0 + prc[i] - askhi0
                askhi[i] = prc[i]

            if (bidlo[i] != 0) and (askhi[i] / bidlo[i] > 8):
                bidlo[i] = np.nan
                askhi[i] = np.nan

        return bidlo, askhi

    def c_bidaskhl_21d(self):
        """High-low bid-ask spread. Corwin and Schultz (2012)"""

        cd = self.data

        bidlo, askhi = self.adjust_bidaskhl(cd.prc_adj.values, cd.bidlo_adj.values, cd.askhi_adj.values, cd.rcount.values)

        prc_l1 = self.shift(cd.prc).values
        lo_l1 = self.remove_rows(np.roll(bidlo, 1))
        hi_l1 = self.remove_rows(np.roll(askhi, 1))

        cond = (prc_l1 < bidlo) & (prc_l1 > 0)
        lo_t = np.where(cond, prc_l1, bidlo)
        hi_t = np.where(cond, askhi - bidlo + prc_l1, askhi)
        cond = (prc_l1 > askhi) & (prc_l1 > 0)
        lo_t = np.where(cond, bidlo + prc_l1 - askhi, lo_t)
        hi_t = np.where(cond, prc_l1, hi_t)

        const = 3 - 2 * 2 ** 0.5
        hi_2d = np.maximum(hi_t, hi_l1)
        lo_2d = np.minimum(lo_t, lo_l1)

        beta = np.where((lo_t > 0) & (lo_l1 > 0),
                        np.log(hi_t / lo_t) ** 2 + np.log(hi_l1 / lo_l1) ** 2, np.nan)
        gamma = np.where(lo_2d > 0, np.log(hi_2d / lo_2d) ** 2, np.nan)
        alpha = (2 ** 0.5 - 1) * np.sqrt(beta) / const - np.sqrt(gamma / const)
        cd['spread'] = np.maximum(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0)
        char = cd.groupby(['permno', 'ym']).spread.mean().swaplevel()
        del cd['spread']

        return char

    def roll_126d(self):
        """Create chars that use 6-month data."""

        if 'ami_126d' in self.chars:  # the chars have already been generated.
            return

        icolumns = ['mon', 'ret', 'dvol', 'vol', 'turnover_d']
        ocolumns = ['ami_126d', 'zero_trades_126d', 'dolvol_126d', 'dolvol_var_126d', 'turnover_126d',
                    'turnover_var_126d']

        @njit(error_model='numpy')
        def fcn(data):
            mons = data[:, 0]
            unique_mons = np.unique(mons)

            ami = np.full((unique_mons.shape[0], 1), np.nan)
            zero = np.full((unique_mons.shape[0], 1), np.nan)
            dolvol = np.full((unique_mons.shape[0], 1), np.nan)
            dolvol_var = np.full((unique_mons.shape[0], 1), np.nan)
            turnover = np.full((unique_mons.shape[0], 1), np.nan)
            turnover_var = np.full((unique_mons.shape[0], 1), np.nan)

            for j, mon in enumerate(unique_mons):
                data_m = data[(mon - 6 < mons) & (mons <= mon), 1:]
                # isnan = isnan1(data_m[:, :1]) | (data_m[:, 1] == 0)
                isnan = isnan1(data_m[:, :1])
                ret = data_m[~isnan, 0]
                dvol = data_m[~isnan, 1]
                if (dvol.shape[0] >= 60) & (dvol != 0).any():  # HXZ require minimum 50 obs, while JKP require 60.
                    ami[j] = 1e6 * (np.abs(ret[dvol != 0]) / dvol[dvol != 0]).mean()
                    dolvol[j] = dvol.mean()
                    dolvol_var[j] = std(dvol) / dolvol[j]

                vol = data_m[:, 2]
                turnover_d = data_m[:, 3]
                turnover_d = turnover_d[~np.isnan(turnover_d)]
                # turnover_d = turnover_d[~(np.isnan(turnover_d) | (turnover_d == 0))]
                if turnover_d.shape[0] >= 60:
                    zero[j] = ((vol == 0).sum() + 1 / turnover_d.sum() / 11000) * 126 / turnover_d.shape[0]
                    turnover[j] = turnover_d.mean()
                    turnover_var[j] = std(turnover_d) / turnover[j]

            return ami, zero, dolvol, dolvol_var, turnover, turnover_var

        # self.chars[ocolumns] = rolling_apply_np(self.data[icolumns], 'permno', fcn)
        retval = rolling_apply_np(self.data[icolumns], 'permno', fcn)
        for c1, c2 in zip(ocolumns, retval):
            self.chars[c1] = c2

    def c_ami_126d(self):
        """Illiquidity. Amihud (2002)"""

        self.roll_126d()
        return self.chars['ami_126d']

    def c_zero_trades_126d(self):
        """Zero-trading days (6 months). Liu (2006)"""

        self.roll_126d()
        return self.chars['zero_trades_126d']

    def c_turnover_126d(self):
        """Share turnover (JKP). Datar, Naik, and Radcliffe (1998)"""

        self.roll_126d()
        return self.chars['turnover_126d']

    def c_turnover_var_126d(self):
        """Volatility of share turnover (JKP). Chordia, Subrahmanyam, and Anshuman (2001)"""

        self.roll_126d()
        return self.chars['turnover_var_126d']

    def c_std_turn(self):
        """Volatility of share turnover (GHZ). Chordia, Subrahmanyam, and Anshuman (2001)"""

        cd = self.data
        cd['_tmp'] = cd.vol / cd.shrout
        return cd.groupby(['permno', 'ym'])._tmp.std().swaplevel()

    def c_dolvol_126d(self):
        """Dollar trading volume (JKP). Brennan, Chordia, and Subrahmanyam (1998)"""

        self.roll_126d()
        return self.chars['dolvol_126d']

    def c_dolvol_var_126d(self):
        """Volatility of dollar trading volume (JKP). Chordia, Subrahmanyam, and Anshuman (2001)"""

        self.roll_126d()
        return self.chars['dolvol_var_126d']

    def c_std_dolvol(self):
        """Volatility of dollar trading volume (GHZ). Chordia, Subrahmanyam, and Anshuman (2001)"""

        cd = self.data
        cd['_tmp'] = np.log(cd.dvol)
        return cd.groupby(['permno', 'ym'])._tmp.std().swaplevel()

    def roll_252d(self):
        """Create chars that use 12-month data."""

        if 'rvol_252d' in self.chars:  # the chars have already been generated.
            return

        icolumns = ['mon', 'exret', 'mktrf', 'prc_adj', 'vol', 'turnover_d']
        ocolumns = ['rvol_252d', 'mktvol_252d', 'ivol_capm_252d', 'prc_highprc_252d', 'zero_trades_252d', 'betadown_252d']

        @njit(error_model='numpy')
        def fcn(data):
            mons = data[:, 0]
            unique_mons = np.unique(mons)

            rvol = np.full((unique_mons.shape[0], 1), np.nan)
            mktvol = np.full((unique_mons.shape[0], 1), np.nan)
            ivol = np.full((unique_mons.shape[0], 1), np.nan)
            prc_high = np.full((unique_mons.shape[0], 1), np.nan)
            zero = np.full((unique_mons.shape[0], 1), np.nan)
            betadown = np.full((unique_mons.shape[0], 1), np.nan)

            for j, mon in enumerate(unique_mons):
                data_m = data[(mon - 12 < mons) & (mons <= mon), 1:]

                # rvol, mktvol, ivol
                retdata = data_m[:, :2]  # exret, mktrf
                retdata = retdata[~isnan1(retdata)]
                if retdata.shape[0] >= 120:
                    endo = retdata[:, 0]
                    exog = retdata[:, 1]

                    rvol[j] = std(endo)
                    mktvol[j] = std(exog)

                    cov = np.cov(endo, exog)
                    beta = cov[0][1] / cov[1][1]
                    c = endo.mean() - beta * exog.mean()
                    pred = c + beta * exog
                    ivol[j] = std(endo - pred)

                # 52-week high
                prc_adj = data_m[:, 2]
                if np.sum(~np.isnan(prc_adj)) >= 120:
                    prc_high[j] = prc_adj[-1] / np.nanmax(prc_adj)
                # prc_adj =  prc_adj[~np.isnan(prc_adj)]
                # if prc_adj.shape[0] >= 120:
                #     prc_high[j] = prc_adj[-1] / prc_adj.max()

                # zero-trades
                vol = data_m[:, 3]
                turnover_d = data_m[:, 4]
                turnover_d = turnover_d[~np.isnan(turnover_d)]
                if turnover_d.shape[0] >= 120:
                    zero[j] = ((vol == 0).sum() + 1 / turnover_d.sum() / 11000) * 252 / turnover_d.shape[0]

                # betadown
                downdata = retdata[retdata[:, 1] < 0]  # mktrf < 0
                if downdata.shape[0] >= 60:
                    cov = np.cov(downdata.T)
                    betadown[j] = cov[0][1] / cov[1][1]

            return rvol, mktvol, ivol, prc_high, zero, betadown

        # self.chars[ocolumns] = rolling_apply_np(self.data[icolumns], 'permno', fcn)
        retval = rolling_apply_np(self.data[icolumns], 'permno', fcn)
        for c1, c2 in zip(ocolumns, retval):
            self.chars[c1] = c2

    def c_rmax5_rvol_21d(self):
        """Highest 5 days of return to volatility. Assness et al. (2020)"""

        self.roll_252d()  # to create rvol_252d
        self.prepare(['rmax5_21d'])
        return self.chars.rmax5_21d / self.chars.rvol_252d

    def c_ivol_capm_252d(self):
        """Idiosyncratic volatility (Org, JKP). Ali, Hwang, and Trombley (2003)"""

        self.roll_252d()
        return self.chars['ivol_capm_252d']

    def c_betadown_252d(self):
        """Downside beta. Ang, Chen, and Xing (2006)"""

        self.roll_252d()
        return self.chars['betadown_252d']

    def c_prc_highprc_252d(self):
        """52-week high. George and Hwang (2004)"""

        self.roll_252d()
        return self.chars['prc_highprc_252d']

    def c_zero_trades_252d(self):
        """Zero-trading days (12 months). Liu (2006)"""

        self.roll_252d()
        return self.chars['zero_trades_252d']

    def corr_1260d(self):
        """Market correlation. Assness et al. (2020)"""

        if 'corr_1260d' in self.chars:
            return

        cd = self.data
        data = cd[['mon']].copy()
        data['exret_3l'] = cd.exret + cd.exret.shift(1) + cd.exret.shift(2)
        data['mktrf_3l'] = cd.mktrf + cd.mktrf.shift(1) + cd.mktrf.shift(2)
        data.loc[cd.rcount < 2, ['exret_3l', 'mktrf_3l']] = np.nan

        @njit(error_model='numpy')
        def fcn(data):
            mons = data[:, 0]
            unique_mons = np.unique(mons)

            corr = np.full((unique_mons.shape[0], 1), np.nan)
            for j, mon in enumerate(unique_mons):
                data_m = data[(mon - 60 < mons) & (mons <= mon), 1:]
                data_m = data_m[~isnan1(data_m)]  # remove nan
                if data_m.shape[0] >= 750:
                    corr[j] = np.corrcoef(data_m.T)[0, 1]

            return corr

        self.chars['corr_1260d'] = rolling_apply_np(data, 'permno', fcn)

    def c_corr_1260d(self):
        """Market correlation. Assness et al. (2020)"""

        self.corr_1260d()
        return self.chars['corr_1260d']

    def c_betabab_1260d(self):
        """Frazzini-Pedersen beta. Frazzini and Pedersen (2014)"""

        self.roll_252d()  # for rvol_252d and mktvol_252d
        self.corr_1260d()
        return self.chars.corr_1260d * self.chars.rvol_252d / self.chars.mktvol_252d

    def reg_weekly_vars(self):
        """Create chars that use weekly returns.
          beta (GHZ), betasq (GHZ), idiovol (GHZ), pricedelay, pricedelay_slope
        """

        if 'pricedelay' in self.chars:
            return

        cd = self.data
        vars = cd[['ret', 'ym']].reset_index()
        vars['yw'] = vars.date.dt.isocalendar().year * 100 + vars.date.dt.isocalendar().week

        vars['wkret'] = np.log(vars.ret + 1)
        gb = vars.groupby(['permno', 'yw'])
        reg = pd.DataFrame(np.exp(gb.wkret.sum()) - 1)
        reg['ym'] = gb.ym.nth(-1)

        gb = reg.groupby(['yw'])
        reg['ewret'] = gb.wkret.transform(np.mean)

        reg['ewret_l1'] = reg['ewret'].shift(1)
        reg['ewret_l2'] = reg['ewret'].shift(2)
        reg['ewret_l3'] = reg['ewret'].shift(3)
        reg['ewret_l4'] = reg['ewret'].shift(4)

        reg['rcount'] = reg.groupby('permno').cumcount()
        reg.loc[reg.rcount < 4, ['ewret_l1', 'ewret_l2', 'ewret_l3', 'ewret_l4']] = np.nan

        reg = reg.swaplevel()

        beta, r2, idio = self.rolling_beta(reg[['wkret', 'ewret']], 156, 52)
        reg['beta'] = beta[:, 1]
        reg['betasq'] = reg['beta'] ** 2
        reg['idiovol'] = idio

        beta2, r2_multi, _ = self.rolling_beta(reg[['wkret', 'ewret', 'ewret_l1', 'ewret_l2', 'ewret_l3', 'ewret_l4']],
                                          156, 52)
        reg['pricedelay'] = 1 - r2 / r2_multi
        reg['pricedelay_slope'] = (beta2[:, 2:] * np.array([1, 2, 3, 4])).sum(1) / beta2[:, 1:].sum(1)

        reg_ym = reg.reset_index().drop_duplicates(subset=['ym', 'permno'], keep='last')
        reg_ym.set_index(['ym', 'permno'], inplace=True)
        reg_ym.index.names = ('date', 'permno')

        self.chars = self.chars.join(reg_ym[['beta', 'betasq', 'idiovol', 'pricedelay', 'pricedelay_slope']],
                                     how='left')

    def c_beta(self):
        """Market beta  (GHZ). Fama and MacBeth (1973)"""

        self.reg_weekly_vars()
        return self.chars['beta']

    def c_betasq(self):
        """Beta squared (GHZ). Fama and MacBeth (1973)"""

        self.reg_weekly_vars()
        return self.chars['betasq']

    def c_idiovol(self):
        """Idiosyncratic volatility (GHZ). Ali, Hwang, and Trombley (2003)"""

        # GHZ use weekly returns.
        self.reg_weekly_vars()
        return self.chars['idiovol']

    def c_pricedelay(self):
        """Price delay based on R-squared. Hou and Moskowitz (2005)"""

        self.reg_weekly_vars()
        return self.chars['pricedelay']

    def c_pricedelay_slope(self):
        """Price delay based on slopes. Hou and Moskowitz (2005)"""

        self.reg_weekly_vars()
        return self.chars['pricedelay_slope']

    def c_baspread(self):
        """Bid-ask spread. Amihud and Mendelson (1986)"""

        cd = self.data
        cd['_tmp'] = (cd.askhi - cd.bidlo) / (0.5 * (cd.askhi + cd.bidlo))
        return cd.groupby(['ym', 'permno'])._tmp.mean().swaplevel()


################################################################################
#
# MERGE
#
################################################################################
class Merge(Panel):
    """Class to generate firm characteristics from a merged dataset of funda, fundq, crspm, and crspd.

    The firm characteristics generated in this class can be viewed using ``Merge.show_available_functions()``:

    >>> Merge().show_available_functions()

    Refer to the manual for usage.
    """

    def __init__(self):
        super().__init__()
        self.data = None

    def preprocess(self, crspm, crspd, funda, fundq, delete_data=True):
        """Merge crspm, crspd, funda, and fundq by left-joining crspd, funda, and fundq to crspm. The resulting data has
        an index=date/permno.

        Args:
            crspm: CRSPM instance
            crspd: CRSPD instance
            funda: FUNDA instance
            fundq: FUNDQ instance
            delete_data: True to delete the data of crspm, crspd, funda, and fundq after merge to save memory.
        """

        elapsed_time('Preprocessing merged...')

        ###############################
        # Prepare characteristics required to generate characteristics here..
        ###############################
        # For age.
        funda.prepare(['age'])

        # For mispricing.
        crspm.prepare(['ret_12_1', 'chcsho_12m', 'eqnpo_12m'])
        funda.prepare(['o_score', 'gp_at', 'oaccruals_at', 'noa_at', 'at_gr1', 'ppeinv_gr1a'])
        fundq.prepare(['niq_at'])

        # For qmj.
        funda.prepare(['gp_at', 'ni_be', 'ocf_at', 'oaccruals_at', 'o_score', 'z_score'])
        fa = funda.data
        fa['ni_at'] = fa.ib / fa.at_
        fa['gp_sale'] = fa.gp / fa.sale
        fa['debt_at'] = fa.debt / fa.at_
        fa['gpoa_ch5'] = funda.diff(fa.gp_at, 5)
        fa['roe_ch5'] = funda.diff(fa.ni_be, 5)
        fa['roa_ch5'] = funda.diff(fa.ni_at, 5)
        fa['cfoa_ch5'] = funda.diff(fa.ocf_at, 5)
        fa['gmar_ch5'] = funda.diff(fa.gp_sale, 5)
        fa['roe_std'] = funda.rolling(fa.ni_be, 5, 'std')

        fq = fundq.data
        fq['roeq_std'] = fundq.rolling(fq.ibq / fq.be, 20, 'std', min_n=12)

        ###############################
        # Merge data
        ###############################
        self.copy_from(crspm)
        if delete_data:
            del crspm.data

        self.merge(crspd.chars, how='left')
        if delete_data:
            del crspd.data
            del crspd.chars

        self.merge(funda.data, on=['date', 'gvkey'], how='left')
        if delete_data:
            del funda.data

        self.merge(fundq.data, on=['date', 'gvkey'], how='left')
        if delete_data:
            del fundq.data

        self.data.sort_index(level=['permno', 'date'], inplace=True)

        # Add target return
        self.data['target'] = self.futret(self.data.exret)
        elapsed_time('merged preprocessed.')

    def postprocess(self):
        """Postprocess data.

        This method deletes temporary variables (variables starting with '\_') and replaces infinite values with nan.
        This method can be overridden to add code to trim or winsorize characteristics.
        """

        elapsed_time('Postprocessing merged...')
        md = self.data

        # drop temporary columns that start with '_'
        del_cols = [col for col in md if col[0] == '_']
        drop_columns(md, del_cols)

        columns = md.columns.intersection(self.get_available_chars())
        inspect_data(md[columns], option=['summary', 'nans'])

        log('Replacing inf with nan...')
        md.replace([np.inf, -np.inf], np.nan, inplace=True)

        elapsed_time('merged postprocessed.')

    #######################################
    # Characteristics
    #######################################
    def c_age(self):
        """Firm age. Jiang, Lee, and Zhang (2005)"""

        # Update age as the maximum of age from funda and age from crspm.
        md = self.data
        age = (md.rcount + 1) * 12 / self.freq  # age from crspm
        return np.maximum(age, md['age'])  # max between crspm age and funda age.

    def c_mispricing_perf(self):
        """Mispricing factor: Performance. Stambaugh and Yuan (2016)"""

        md = self.data
        tmp = md.groupby(level=0)[['o_score', 'ret_12_1', 'gp_at', 'niq_at']].rank(pct=True)
        tmp.o_score = 1 - tmp.o_score
        return tmp.mean(axis=1)

    def c_mispricing_mgmt(self):
        """Mispricing factor: Management. Stambaugh and Yuan (2016)"""

        md = self.data
        tmp = md.groupby(level=0)[['chcsho_12m', 'eqnpo_12m', 'oaccruals_at', 'noa_at', 'at_gr1', 'ppeinv_gr1a']].rank(
            ascending=False, pct=True)
        if REPLICATE_JKP:
            tmp.eqnpo_12m = 1 - tmp.eqnpo_12m
        char = np.where(tmp.count(axis=1) < 3, np.nan, tmp.mean(axis=1))
        return char

    @staticmethod
    def zrank(chars):
        """Normalized rank.
        """

        def zrank_(x):
            rx = x.rank()
            y = (rx - rx.mean()) / rx.std()
            if x.ndim == 1:
                return y

            ry = y.mean(axis=1).rank()
            return (ry - ry.mean()) / ry.std()

        return groupby_apply(chars.groupby(level=0), zrank_)


    def c_qmj_growth(self):
        """Quality minus Junk: Growth. Assness, Frazzini, and Pedersen (2018)"""

        if 'qmj_growth' in self.data:
            return self.data['qmj_growth']

        md = self.data

        grow = md[['gpoa_ch5', 'roe_ch5', 'roa_ch5', 'cfoa_ch5', 'gmar_ch5']]
        return self.zrank(grow)

    def c_qmj_prof(self):
        """Quality minus Junk: Profitability. Assness, Frazzini, and Pedersen (2018)"""

        if 'qmj_prof' in self.data:
            return self.data['qmj_prof']

        md = self.data

        prof = md[['gp_at', 'ni_be', 'ni_at', 'ocf_at', 'gp_sale']].copy()
        prof['oaccruals_at'] = -md.oaccruals_at
        return self.zrank(prof)

    def c_qmj_safety(self):
        """Quality minus Junk: Safety. Assness, Frazzini, and Pedersen (2018)"""

        if 'qmj_safety' in self.data:
            return self.data['qmj_safety']

        md = self.data

        safe = -md[['betabab_1260d', 'debt_at', 'o_score']].copy()
        safe['z_score'] = md.z_score
        safe['evol'] = -md['roeq_std'].fillna(md['roe_std'] / 2)

        return self.zrank(safe)

    def c_qmj(self):
        """Quality minus Junk: Composite. Assness, Frazzini, and Pedersen (2018)"""

        md = self.data

        self.prepare(['qmj_growth', 'qmj_prof', 'qmj_safety'])
        qmj = (md['qmj_prof'] + md['qmj_growth'] + md['qmj_safety']) / 3
        return self.zrank(qmj)


if __name__ == '__main__':
    os.chdir('../')

