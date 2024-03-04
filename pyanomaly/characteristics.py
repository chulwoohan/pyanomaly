"""This module defines classes for firm characteristic generation.

    .. autosummary::
        :nosignatures:

        FUNDA
        FUNDQ
        CRSPM
        CRSPDRaw
        CRSPD
        Merge
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


################################################################################
#
# FUNDA
#
################################################################################

class FUNDA(FCPanel):
    """Class to generate firm characteristics from funda.

    The firm characteristics defined in this class can be viewed using :meth:`.show_available_chars`.

    Args:
        alias (str, list, or dict): Firm characteristics to generate and their aliases. If None, all available firm
            characteristics are generated.

            * str: A column name in the mapping file (``config.mapping_file_path``). The firm characteristics defined
              in this class and in the `alias` column of the mapping file are generated. See :ref:`sec-mapping file`.
            * list: List of firm characteristics (method names).
            * dict: Dict of firm characteristics and their aliases of the form {method name: alias}.

            If aliases are not specified, method names are used as aliases.
        data: DataFrame of funda data with index = datadate/gvkey, sorted on gvkey/datadate. The funda data can be
            given at initialization or loaded later using :meth:`load_data`.

    **Methods**

    .. autosummary::
        :nosignatures:

        load_data
        convert_currency
        merge_with_fundq
        add_crsp_me
        update_variables
        postprocess

    Firm characteristic generation methods have a name like ``c_characteristic()``.
    """

    def __init__(self, alias=None, data=None):
        super().__init__(alias, data, ANNUAL)

    def load_data(self, sdate=None, edate=None, fname='funda'):
        """Load funda data from file.

        This method loads funda data from ``config.input_dir/comp/funda`` and stores it in the ``data``
        attribute. The ``data`` has index = datadate/gvkey and is sorted on gvkey/datadate.

        Args:
            sdate: Start date ('yyyy-mm-dd'). If None, from the earliest.
            edate: End date ('yyyy-mm-dd'). If None, to the latest.
            fname: The funda file name.
        """

        elapsed_time(f'Loading {fname}...')
        data = WRDS.read_data(fname, index_col=['datadate', 'gvkey'])
        if not self.is_sorted(data):
            data.sort_index(level=[-1, 0], inplace=True)

        if sdate:
            data = data[data.index.get_level_values(0) >= sdate]
        if edate:
            data = data[data.index.get_level_values(0) <= edate]

        self.data = data
        elapsed_time(f'{fname} loaded.')
        self.inspect_data()

    # def cleanse_data(self, method='Quantile', **kwargs):
    #     data = self.data.copy()
    #
    #     # Columns to examine.
    #     excl_columns = ['fyear']
    #     columns = [col for col in data.columns if is_numeric(data[col]) and (col not in excl_columns)]
    #     keep_columns(data, columns)
    #
    #     # Adjust for splits.
    #     data['prcc_f'] /= data.ajex
    #     data['csho'] *= data.ajex
    #
    #     anomaly = detect_anomalies(data, None, method, **kwargs)
    #     for i, col in enumerate(columns):
    #         self.data.loc[anomaly[:, i], col] = np.nan

    def convert_currency(self):
        """Convert currency to USD.

        Convert the currency of funda to USD. This method needs to be called if

            i) the data contains non USD-denominated firms, e.g., CAD; and
            ii) CRSP's market equity is used, which is always in USD.

        See Also:
            :meth:`.WRDS.convert_fund_currency_to_usd`
        """

        elapsed_time(f'Converting non USD items to USD...')
        if (self.data['curcd'] != 'USD').any():
            self.data = WRDS.convert_fund_currency_to_usd(self.data, table='funda')
        elapsed_time(f'Currency converted.')

    def merge_with_fundq(self, fundq):
        """Merge funda with fundq.

        Merge funda with quarterly-updated annual data generated from fundq. If a value exists in both data,
        funda has the priority.

        NOTE:
            JKP create characteristics in funda and fundq separately and merge them, whereas we merge the raw data
            first and then generate characteristics. Since some variables in funda are not available in fundq, e.g.,
            ebitda, JKP synthesize those unavailable variables with other variables and create characteristics, even when
            they are available in funda. We prefer to merge funda with fundq at the raw data level and create
            characteristics from the merged data.

            Columns in both funda and fundq:

                datadate, cusip, cik, sic, naics, sale, revt, cogs, xsga, dp, xrd, ib, nopi, spi, pi, txp, ni, txt, xint,
                capx, oancf, gdwlia, gdwlip, rect, act, che, ppegt, invt, at, aco, intan, ao, ppent, gdwl, lct, dlc, dltt,
                lt, pstk, ap, lco, lo, drc, drlt, txdi, ceq, scstkc, csho, prcc_f, oibdp, oiadp, mii, xopr, xi, do, xido,
                ibc, dpc, xidoc, fincf, fiao, txbcof, dltr, dlcch, prstkc, sstk, dv, ivao, ivst, re, txditc, txdb, seq,
                mib, icapt, ajex, curcd, exratd

            Columns in funda but not in fundq:

                xad, gp, ebitda, ebit, txfed, txfo, dvt, ob, gwo, fatb, fatl, dm, dcvt, cshrc, dcpstk, emp, xlr, ds, dvc,
                itcb, pstkrv, pstkl, dltis, ppenb, ppenls

        Args:
            fundq: :class:`FUNDQ` instance.
        """

        elapsed_time('Merging funda with fundq...')
        fa = self.data
        fq = fundq.generate_funda_vars()  # quarterly updated annual data

        if config.replicate_jkp:
            for col in fa:
                if col not in fq:
                    fq[col] = np.nan

        common_columns = list(fq.columns.intersection(fa.columns))  # columns in both funda and fundq
        common_columns_q = [col + '_q' for col in common_columns]
        fq = fq.loc[:, common_columns]

        # Merge funda with fundq. fundq.X changes to fundq.X_q
        fa = fa.merge(fq, on=['date', 'gvkey'], how='outer', suffixes=['', '_q']).sort_index(level=['gvkey', 'date'])

        # Replace X with X_q
        cond = fa['datadate'].isna() | (fa['datadate'] < fa['datadate_q'])
        fa['datadate'] = np.where(cond, fa['datadate_q'], fa['datadate'])

        for col, qcol in zip(common_columns, common_columns_q):
            if col == 'datadate':
                continue
            if config.replicate_jkp:
                # JKP prioritize fundq: fundq will replace funda if datadate < datadate_q even when funda has values
                # and fundq is missing.
                fa[col].mask(cond, fa[qcol], inplace=True)
            else:
                # Check missing values column by column and use fundq only when fundq is not missing.
                cond1 = (cond | fa[col].isna()) & ~fa[qcol].isna()
                # cond = fa[col].isna() | (~fa[qcol].isna() & (fa['datadate'] < fa['datadate_q']))
                fa[col] = fa[col].mask(cond1, fa[qcol])

        drop_columns(fa, common_columns_q)
        for col in fa:  # To release memory
            fa[col] = fa[col]

        self.data = fa
        elapsed_time('funda and fundq merged.')

    def add_crsp_me(self, crspm, method='latest'):
        """Add crsp market equity.

        In funda, market equity ('me') and fiscal market equity ('me_fiscal') are both defined as (prcc_f * csho).
        This method replaces them with crspm's firm-level market equity ('me_company').
        If `method` = 'latest', 'me' is the latest 'me_company' and 'me_fiscal' is the 'me_company' on datadate.
        If `method` = 'year_end', both 'me' and 'me_fiscal' are the 'me_company' in December of datadate year.

        Args:
            crspm: :class:`CRSPM` instance.
            method: How to merge crsp me with funda. 'latest': latest me; 'year_end': December me.
        """

        elapsed_time('Adding crspm me to funda...')
        cm = crspm.data
        cm = cm.loc[(cm.primary == True) & (cm.me_company > 0) & (~cm.gvkey.isna()), ['gvkey', 'me_company']]
        cm.rename(columns={'me_company': 'me'}, inplace=True)

        fa = self.data
        drop_columns(fa, ['me', 'me_fiscal'])  # remove me from funda

        if method == 'latest':
            fa = merge(fa, cm, on=['date', 'gvkey'], how='left')

            # Add me_fiscal
            tmp = fa['me'].reset_index().rename(columns={'date': 'datadate', 'me': 'me_fiscal'})
            fa = merge(fa, tmp, on=['gvkey', 'datadate'], how='left')
        elif method == 'year_end':
            cm.reset_index(inplace=True)
            cm = cm[cm.date.dt.month == 12]
            cm['year'] = cm.date.dt.year

            fa['year'] = fa.datadate.dt.year
            fa = merge(fa, cm[['year', 'gvkey', 'me']], on=['year', 'gvkey'], how='left')
            fa['me_fiscal'] = fa['me']
            del fa['year']
        else:
            raise ValueError(f"Methods to merge crsp me with funda: ['latest', 'year_end']. '{method}' is given.")

        self.data = fa
        elapsed_time(f'crspm me added to funda. funda shape: {fa.shape}')

    def update_variables(self):
        """Preprocess data before creating firm characteristics.

        1. Synthesize missing values with other variables.
        2. Create frequently used variables.
        """

        elapsed_time(f'updating funda variables...')

        fa = self.data

        # Cleansing
        # non-negative variables with a negative value -> set to nan.
        for var in ['at', 'sale', 'revt', 'dv', 'che']:
            fa.loc[fa[var] < 0, var] = np.nan  # should be >= 0.

        # tiny values -> set to 0.
        for col in fa.columns:
            if is_float(fa[col]):
                fa.loc[is_zero(fa[col]), col] = 0

        fa.rename(columns={'at': 'at_', 'lt': 'lt_'}, inplace=True)  # 'at', 'lt' are reserved names in python.

        fa['ivao'] = fa.ivao.fillna(0)
        fa['ivst'] = fa.ivst.fillna(0)

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
        fa['txditc'] = fa.txditc.fillna(nansum1(fa.txdb, fa.itcb))  # used in 'at' before this conversion

        # Balance sheet - Financing
        fa['debt'] = nansum1(fa.dltt, fa.dlc)
        fa['be'] = fa.seq + fa.txditc.fillna(0) - fa.pstkrv.fillna(0)
        fa['bev'] = (fa.icapt + fa.dlc.fillna(0) - fa.che.fillna(0)).fillna(
            fa.seq + fa.debt - fa.che.fillna(0) + fa.mib.fillna(0))

        # Market equities
        fa['me'] = fa.prcc_f * fa.csho
        fa['me_fiscal'] = fa['me']

        # Accruals
        cowc = fa.act - fa.che - (fa.lct - fa.dlc.fillna(0))
        nncoa = fa.at_ - fa.act - fa.ivao - (fa.lt_ - fa.lct - fa.dltt)
        fa['oacc'] = (fa.ib - fa.oancf).fillna(self.diff(cowc + nncoa))
        # EDIT: 2023.07.01. The following line has been deleted.
        # fa.loc[(fa.ib.isna() | fa.oancf.isna()) & (fa.rcount < 12), 'oacc'] = np.nan  # if diff(cowc + nncoa) is used, the
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

        This method deletes temporary variable columns (columns starting with '\\_') and replaces infinite
        values with nan.
        """

        elapsed_time('Postprocessing funda...')
        fa = self.data

        # Drop temporary columns that start with '_'
        del_cols = [col for col in fa if col[0] == '_']
        drop_columns(fa, del_cols)

        char_cols = list(self.get_char_list())
        char_cols = list(fa.columns.intersection(char_cols))
        self.inspect_data(char_cols, option=['summary', 'nans'])

        log('Replacing inf with nan...')
        for col in char_cols:
            isinf = (fa[col] == np.inf) | (fa[col] == -np.inf)
            if isinf.any():
                fa.loc[isinf, col] = np.nan

        self.inspect_data(char_cols, option=['stats'])
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

        if config.replicate_jkp:
            char = self.pct_change(fa.debt, 3)  # JKP
        else:
            char = self.pct_change(fa.debt, 5)
            # Below is the original definition but using log causes inf when either debt of debt_l5 is 0.
            # Therefore, we use pct_change, which results in inf only when debt_l5 is 0.
            # char = np.log(fa.debt / self.shift(fa.debt, 5))
        return char

    def c_inv_gr1a(self):
        """Inventory change. Thomas and Zhang (2002)"""

        fa = self.data
        if config.replicate_jkp:
            char = self.diff(fa.invt) / fa.at_  # JKP use at at t.
        else:
            char = self.diff(fa.invt) / fa.avgat
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
        char = self.diff(fa.at_ - fa.act - fa.ivao) / fa.at_
        return char

    def c_ncol_gr1a(self):
        """Change in non-current operating liabilities. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.lt_ - fa.lct - fa.dltt) / fa.at_
        return char

    def c_nncoa_gr1a(self):
        """Change in net non-current operating assets. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.at_ - fa.act - fa.ivao - (fa.lt_ - fa.lct - fa.dltt)) / fa.at_
        return char

    def c_noa_gr1a(self):
        """Change in net operating assets. Hirshleifer et al. (2004)"""

        fa = self.data
        noa = fa.act - fa.che + fa.at_ - fa.act - fa.ivao - (
                fa.lct - fa.dlc.fillna(0) + (fa.lt_ - fa.lct - fa.dltt))
        char = self.diff(noa) / self.shift(fa.at_)
        return char

    def c_fnl_gr1a(self):
        """Change in financial liabilities. Richardson et al. (2005)"""

        fa = self.data
        fnl = fa.debt + fa.pstkrv
        if config.replicate_jkp:
            char = self.diff(fnl) / fa.at_  # jkp uses 'at' at t while hxz uses 'at' at t-1.
        else:
            char = self.diff(fnl) / fa.avgat  # jkp uses 'at' at t while hxz uses 'at' at t-1.
        return char

    def c_nfna_gr1a(self):
        """Change in net financial assets. Richardson et al. (2005)"""

        fa = self.data
        char = self.diff(fa.ivst + fa.ivao - (fa.debt + fa.pstkrv)) / fa.at_
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
        char[fa.bev < ZERO] = np.nan
        return char

    def c_netis_at(self):
        """Net external finance. Bradshaw, Richardson, and Sloan (2006)"""

        # nxf = nef + ndf
        # nef = sstk - prstkc - dv, ndf = dltis - dltr + dlcch
        fa = self.data
        ddltt = self.diff(fa.dltt)
        ddlc = self.diff(fa.dlc)
        dbnetis = nansum1(nansum1(fa.dltis, -fa.dltr).fillna(ddltt), fa.dlcch.fillna(ddlc))
        if config.replicate_jkp:
            netis = nansum1(fa.sstk, -fa.prstkc) + dbnetis
        else:
            netis = nansum1(fa.sstk, -fa.prstkc, -fa.dv) + dbnetis
        return netis / fa.at_

    def c_eqnetis_at(self):
        """Net equity finance. Bradshaw, Richardson, and Sloan (2006)"""

        fa = self.data
        if config.replicate_jkp:
            return nansum1(fa.sstk, -fa.prstkc) / fa.at_
        else:
            return nansum1(fa.sstk, -fa.prstkc, -fa.dv) / fa.at_

    def c_dbnetis_at(self):
        """Net debt finance. Bradshaw, Richardson, and Sloan (2006)"""

        fa = self.data
        ddltt = self.diff(fa.dltt)
        ddlc = self.diff(fa.dlc)
        dbnetis = nansum1(nansum1(fa.dltis, -fa.dltr).fillna(ddltt), fa.dlcch.fillna(ddlc))
        return dbnetis / fa.at_

    def c_oaccruals_at(self):
        """Operating Accruals (JKP). Sloan (1996)"""

        fa = self.data
        if config.replicate_jkp:
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
        tacc = fa.oacc + self.diff(fa.ivst + fa.ivao - (fa.debt + fa.pstkrv))
        char = tacc / fa.at_
        return char

    def c_taccruals_ni(self):
        """Percent total accruals. Hafzalla, Lundholm, and Van Winkle (2011)"""

        fa = self.data
        tacc = fa.oacc + self.diff(fa.ivst + fa.ivao - (fa.debt + fa.pstkrv))
        char = tacc / fa.nix.abs()
        char[fa.nix == 0] = np.nan
        return char

    def c_noa_at(self):
        """Net operating assets. Hirshleifer et al. (2004)"""

        fa = self.data
        noa = fa.act - fa.che + fa.at_ - fa.act - fa.ivao - (
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
        char[fa.bev < ZERO] = np.nan
        return char

    def c_rd_sale(self):
        """R&D to sales. Chan, Lakonishok, and Sougiannis (2001) (Guo, Lev, and Shi (2006) in GHZ)"""

        fa = self.data
        char = fa.xrd / fa.sale
        char[fa.sale < ZERO] = np.nan
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
        return (fa.dvt - nansum1(fa.sstk, -fa.prstkc)) / fa.me

    def c_ebitda_mev(self):
        """Enterprise multiple. Loughran and Wellman (2011)"""

        fa = self.data

        mev = fa.me + fa.debt - fa.che.fillna(0)
        char = fa.ebitda / mev
        char[mev < ZERO] = np.nan

        return char

    def c_enterprise_multiple(self):
        """Enterprise multiple. Loughran and Wellman (2011)"""
        # This is the correct definition of enterprise multiple.

        fa = self.data

        char = (fa.me + fa.debt + fa.pstkrv - fa.che.fillna(0)) / fa.oibdp
        char[fa.oibdp < ZERO] = np.nan

        return char

    def c_ocf_at(self):
        """Operating cash flow to assets. Bouchard et al. (2019)"""

        fa = self.data
        char = fa.ocf / fa.at_
        return char

    def c_ocf_at_chg1(self):
        """Change in operating cash flow to assets. Bouchard et al. (2019)"""

        self.prepare(['ocf_at'])
        fa = self.data
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
        if config.replicate_jkp:
            lnoa = fa.ppent + fa.intan + fa.ao - fa.lo + fa.dp
            char = self.diff(lnoa) / (fa.at_ + self.shift(fa.at_))
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
        return char

    def c_rd5_at(self):
        """R&D capital-to-assets. Li (2011)"""

        fa = self.data
        char = (fa.xrd + 0.8 * self.shift(fa.xrd) + 0.6 * self.shift(fa.xrd, 2) +
                0.4 * self.shift(fa.xrd, 3) + 0.2 * self.shift(fa.xrd, 4)) / fa.at_
        return char

    def c_age(self):
        """Firm age. Jiang, Lee, and Zhang (2005)"""

        return (self.get_row_count() + 1) * 12 / self.freq

    def _chg_to_exp(self, char):
        if config.replicate_jkp:
            exp = self.rolling(char, 2, 'mean', min_n=2, lag=1)  # JKP require both lagged values.
        else:
            exp = self.rolling(char, 2, 'mean', min_n=1,
                               lag=1)  # min_n=1: calculate if there is at least one observation.
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
        if config.replicate_jkp:
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
        char[fa.ceq < ZERO] = np.nan
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
        if config.replicate_jkp:
            f_lev = self.diff(fa.dltt / fa.at_)  # jkp use at
            f_eqis = fa.sstk  # jkp omit pstk
        else:
            f_lev = self.diff(fa.dltt / fa.avgat)  # jkp use at
            f_eqis = (fa.sstk - fa.pstk).fillna(0)  # jkp omit pstk

        char = (f_roa > ZERO).astype(int) + (f_croa > ZERO).astype(int) + (f_droa > ZERO).astype(int) \
               + (f_acc > ZERO).astype(int) + (f_lev < ZERO).astype(int) + (f_liq > ZERO).astype(int) \
               + is_zero(f_eqis).astype(int) + (f_gm > ZERO).astype(int) + (f_aturn > ZERO).astype(int)
        return char.astype(config.float_type)

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
        return char.astype(config.float_type)

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
        o_cacl[fa.act < ZERO] = np.nan
        o_ffo = (fa.pi + fa.dp) / fa.lt_
        o_ffo[fa.lt_ < ZERO] = np.nan
        o_neg_eq = fa.lt_ > fa.at_
        o_neg_earn = (nix < 0) & (nix_1 < 0)
        denom = np.abs(nix) + np.abs(nix_1)
        # denom = nix.abs() + nix_1.abs()
        o_nich = (nix - nix_1) / denom
        o_nich[denom < ZERO] = np.nan

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
        if config.replicate_jkp:
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
        char[fa.emp < ZERO] = np.nan
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
        char[denom < ZERO] = np.nan
        return char

    def _ni_ar1_ivol(self):
        """Earnings persistence and predictability"""

        if self.created('ni_ar1'):
            return

        fa = self.data
        n = 5  # window size

        reg = pd.DataFrame(index=fa.index)
        reg['ni_at'] = fa.ib / fa.at_
        reg['ni_at_lag1'] = self.shift(reg.ni_at)

        res = self.rolling_regression(reg, n)
        fa['ni_ar1'] = res[:, 1]
        fa['ni_ivol'] = res[:, -1]

    def c_ni_ar1(self):
        """Earnings persistence. Francis et al. (2004)"""

        self._ni_ar1_ivol()
        return self.data['ni_ar1']

    def c_ni_ivol(self):
        """Earnings predictability. Francis et al. (2004)"""

        self._ni_ar1_ivol()
        return self.data['ni_ivol']

    ####################################
    # Only in GHZ
    ####################################
    def c_cashpr(self):
        """Cash productivity. Chandrashekar and Rao (2009)"""

        fa = self.data
        char = (fa.me + fa.dltt - fa.at_) / fa.che
        char[fa.che < ZERO] = np.nan
        return char

    def c_roic(self):
        """Return on invested capital. Brown and Rowe (2007)"""

        fa = self.data
        char = (fa.ebit - fa.nopi) / (fa.ceq + fa.lt_ - fa.che)
        char[fa.ceq + fa.lt_ - fa.che < ZERO] = np.nan
        return char

    def c_absacc(self):
        """Absolute accruals. Bandyopadhyay, Huang, and Wirjanto (2010)"""

        self.prepare(['acc'])
        fa = self.data
        return fa.acc.abs()

    def c_depr(self):
        """Depreciation to PP&E. Holthausen and Larcker (1992)"""

        fa = self.data
        char = fa.dp / fa.ppent
        char[fa.ppent < ZERO] = np.nan
        return char

    def c_pchdepr(self):
        """Change in depreciation to PP&E. Holthausen and Larcker (1992)"""

        fa = self.data
        char = self.pct_change(fa.dp / fa.ppent)
        char[is_zero(fa.ppent)] = np.nan
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
        return char.astype(config.float_type)

    def c_currat(self):
        """Current ratio. Ou and Penman (1989)"""

        fa = self.data
        act = fa.act.fillna(fa.che + fa.rect + fa.invt)
        lct = fa.lct.fillna(fa.ap)
        char = act / lct
        char[is_zero(lct)] = np.nan
        return char

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
        char = (act - fa.invt) / lct
        char[is_zero(lct)] = np.nan
        return char

    def c_pchquick(self):
        """Change in quick ratio. Ou and Penman (1989)"""

        self.prepare(['quick'])
        fa = self.data
        char = self.pct_change(fa.quick)
        return char

    def c_salecash(self):
        """Sales-to-cash. Ou and Penman (1989)"""

        fa = self.data
        char = fa.sale / fa.che
        char[fa.che < ZERO] = np.nan
        return char

    def c_salerec(self):
        """Sales-to-receivables. Ou and Penman(1989)"""

        fa = self.data
        char = fa.sale / fa.rect
        char[fa.rect < ZERO] = np.nan
        return char

    def c_saleinv(self):
        """Sales-to-inventory. Ou and Penman(1989)"""

        fa = self.data
        char = fa.sale / fa.invt
        char[fa.invt < ZERO] = np.nan
        return char

    def c_pchsaleinv(self):
        """Change in sales to inventory. Ou and Penman(1989)"""

        self.prepare(['saleinv'])
        fa = self.data
        char = self.pct_change(fa.saleinv)
        return char

    def c_cashdebt(self):
        """Cash flow-to-debt. Ou and Penman(1989)"""

        fa = self.data
        denom = self.rolling(fa.lt_, 2, 'mean')
        char = (fa.ib + fa.dp) / denom
        char[denom < ZERO] = np.nan
        return char

    def c_rd(self):
        """Unexpected R&D increase. Eberhart, Maxwell, and Siddique (2004)"""

        fa = self.data
        char = ((fa.xrd / fa.revt >= 0) & (fa.xrd / fa.at_ >= 0) &
                (self.pct_change(fa.xrd) >= 0.05) & (self.pct_change(fa.xrd / fa.at_) >= 0.05))
        return char.astype(config.float_type)

    def c_chpmia(self):
        """Change in profit margin. Soliman (2008)"""

        fa = self.data
        ib_sale = fa.ib / fa.sale
        ib_sale[fa.sale < ZERO] = np.nan
        char = self.diff(ib_sale)
        # ghz
        # - ghz adjust for industry.
        # fa['_chpm'] = self.diff(fa.ib / fa.sale)
        # char = fa['_chpm'] - fa.groupby(['sic2', 'fyear'])['_chpm'].transform('mean')
        return char

    def c_chatoia(self):
        """Change in profit margin. Soliman (2008)"""

        fa = self.data
        opat = fa.rect + fa.invt + fa.aco + fa.ppent + fa.intan - fa.ap - fa.lco - fa.lo
        opat_avg = self.rolling(opat, 2, 'mean')
        opat_avg[is_zero(opat_avg)] = np.nan
        char = self.diff(fa.sale / opat_avg)
        # ghz
        # - ghz use avg(at) and adjust for industry.
        # fa['_chato'] = self.diff(fa.sale / fa.avgat)  # ghz
        # char = fa['_chato'] - fa.groupby(['sic2', 'fyear'])['_chato'].transform('mean')
        return char

    def c_bm_ia(self):
        """Industry-adjusted book-to-market. Asness, Porter, and Stevens (2000)"""

        fa = self.data
        fa['_tmp'] = fa.ceq / fa.me  # bm
        return fa['_tmp'] - fa.groupby(['sic2', 'fyear'])['_tmp'].transform('mean')

    def c_cfp_ia(self):
        """Industry-adjusted cash flow-to-price. Asness, Porter, and Stevens (2000)"""

        self.prepare(['cfp'])
        fa = self.data
        return fa.cfp - fa.groupby(['sic2', 'fyear']).cfp.transform('mean')

    def c_chempia(self):
        """Industry-adjusted change in employees. Asness, Porter, and Stevens (2000)"""

        fa = self.data
        fa['_tmp'] = self.pct_change(fa.emp).fillna(0)
        return fa['_tmp'] - fa.groupby(['sic2', 'fyear'])['_tmp'].transform('mean')

    def c_mve_ia(self):
        """Industry-adjusted firm size. Asness, Porter, and Stevens (2000)"""

        fa = self.data
        return fa.me - fa.groupby(['sic2', 'fyear']).me.transform('mean')

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
        char = fa.secured > 0
        return char.astype(config.float_type)

    def c_secured(self):
        """Secured debt-to-total debt. Valta (2016)"""

        fa = self.data
        char = fa.dm / (fa.dltt + fa.dlc)
        char[fa.dltt + fa.dlc < ZERO] = np.nan
        return char.fillna(0)

    def c_convind(self):
        """Convertible debt indicator. Valta (2016)"""

        fa = self.data
        dc = fa.dcvt.fillna(fa.dcpstk - fa.pstk.fillna(0))
        dc[dc < ZERO] = np.nan
        char = (dc > 0) | (fa.cshrc > 0)
        return char.astype(config.float_type)

    def c_pchcapx_ia(self):
        """Industry-adjusted change in capital investment. Abarbanell and Bushee (1998)"""

        fa = self.data
        capx = fa.capx.fillna(self.diff(fa.ppent))
        fa['_pchcapx'] = self._chg_to_exp(capx)
        char = fa['_pchcapx'] - fa.groupby(['sic2', 'fyear'])['_pchcapx'].transform('mean')
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
        char[(numer > 0) & (fa.ib <= ZERO)] = 1
        char[(numer <= 0) & (fa.ib <= ZERO)] = np.nan
        return char

    def _herf(self, char):
        # Industry concentration. Hou and Robinson (2006)

        fa = self.data
        fa['_sic3'] = fa.sic.str[:3]  # first three digit
        gb = fa.groupby(['_sic3', 'fyear'])

        fa['_tmp'] = (fa[char] / gb[char].transform('sum')) ** 2
        char = gb['_tmp'].transform('sum')

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
class FUNDQ(FCPanel):
    """Class to generate firm characteristics from fundq.

    The firm characteristics defined in this class can be viewed using :meth:`.show_available_chars`.

    Args:
        alias (str, list, or dict): Firm characteristics to generate and their aliases. If None, all available firm
            characteristics are generated.

            * str: A column name in the mapping file (``config.mapping_file_path``). The firm characteristics defined
              in this class and in the `alias` column of the mapping file are generated. See :ref:`sec-mapping file`.
            * list: List of firm characteristics (method names).
            * dict: Dict of firm characteristics and their aliases of the form {method name: alias}.

            If aliases are not specified, method names are used as aliases.
        data: DataFrame of fundq data with index = datadate/gvkey, sorted on gvkey/datadate. The fundq data can be
            given at initialization or loaded later using :meth:`load_data`.

    **Methods**

    .. autosummary::
        :nosignatures:

        load_data
        remove_duplicates
        convert_currency
        create_qitems_from_yitems
        update_variables
        postprocess

    Firm characteristic generation methods have a name like ``c_characteristic()``.
    """

    def __init__(self, alias=None, data=None):
        super().__init__(alias, data, QUARTERLY)

    def load_data(self, sdate=None, edate=None, fname='fundq'):
        """Load fundq data from file.

        This method loads fundq data from ``config.input_dir/comp/fundq`` and stores it in the ``data``
        attribute. The ``data`` has index = datadate/gvkey and is sorted on gvkey/datadate.

        Args:
            sdate: Start date ('yyyy-mm-dd'). If None, from the earliest.
            edate: End date ('yyyy-mm-dd'). If None, to the latest.
            fname: The fundq file name.
        """

        elapsed_time(f'Loading {fname}...')
        data = WRDS.read_data(fname, index_col=['datadate', 'gvkey'])
        if not self.is_sorted(data):
            data.sort_index(level=[-1, 0], inplace=True)

        if sdate:
            data = data[data.index.get_level_values(0) >= sdate]
        if edate:
            data = data[data.index.get_level_values(0) <= edate]

        self.data = data
        elapsed_time(f'{fname} loaded.')
        self.inspect_data()

    # def cleanse_data(self, method='Quantile', **kwargs):
    #     data = self.data.copy()
    #
    #     # Columns to examine.
    #     excl_columns = ['fyr', 'fyearq', 'fqtr']
    #     columns = [col for col in data.columns if is_numeric(data[col]) and (col not in excl_columns)]
    #     keep_columns(data, columns)
    #
    #     # Adjust for splits.
    #     data['prccq'] /= data.ajexq
    #     data['cshoq'] *= data.ajexq
    #
    #     anomaly = detect_anomalies(data, None, method, **kwargs)
    #     for i, col in enumerate(columns):
    #         self.data.loc[anomaly[:, i], col] = np.nan

    def remove_duplicates(self):
        """Drop duplicates.

        In fundq, there are duplicate rows (rows with the same datadate and gvekey).
        Remove duplicates in the following order:

            1. Remove records with missing fqtr.
            2. Choose records with the maximum fyearq.
            3. Choose records with the minimum fqtr.
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
        """Convert currency to USD.

        Convert the currency of fundq to USD. This method needs to be called if

            i) the data contains non USD-denominated firms, e.g., CAD; and
            ii) CRSP's market equity is used, which is always in USD.

        See Also:
            :meth:`.WRDS.convert_fund_currency_to_usd`
        """

        elapsed_time(f'Converting non USD items to USD...')
        if (self.data.curcdq != 'USD').any():
            self.data = WRDS.convert_fund_currency_to_usd(self.data, table='fundq')
        elapsed_time(f'Currency converted.')

    def _get_ytd_columns(self):
        """Get year-to-data columns."""

        return [col for col in self.data.columns if (col[-1] == 'y') and (col != 'gvkey')]

    def create_qitems_from_yitems(self):
        """Quarterize ytd items.

        Quarterize ytd variables, Xy's, and use them to fill missing Xq's (if Xq exists) or to
        create new quarterly variables (if Xq does not exist).
        """

        elapsed_time('Creating quarterly items from ytd items...')

        fq = self.data
        ycolumns = self._get_ytd_columns()
        qtr_data = self.diff(fq[ycolumns])
        qtr_data[fq.fqtr == 1] = fq.loc[fq.fqtr == 1, ycolumns].values

        for col in ycolumns:
            qcol = col[:-1] + 'q'
            if qcol in fq.columns:
                fq[qcol] = fq[qcol].fillna(qtr_data[col])
            else:  # create new q-item.
                fq[qcol] = qtr_data[col]

        self.data = fq
        elapsed_time('Quarterly items created from ytd items.')

    def update_variables(self):
        """Preprocess data before creating firm characteristics.

        1. Synthesize missing values with other variables.
        2. Create frequently used variables.
        """

        elapsed_time(f'Updating fundq variables...')

        fq = self.data

        # Cleansing
        # non-negative variables with a negative value -> set to nan.
        for var in ['atq', 'saleq', 'saley', 'revtq', 'cheq']:
            fq.loc[fq[var] < 0, var] = np.nan  # should be >= 0.

        seq = fq.seqq.fillna(fq.ceqq + fq.pstkq).fillna(fq.atq - fq.ltq)
        fq['be'] = seq + fq.txditcq.fillna(0) - fq.pstkq.fillna(0)

        # tiny values -> set to 0.
        for col in fq.columns:
            if is_float(fq[col]):
                fq.loc[is_zero(fq[col]), col] = 0
        # for col in fq.columns[is_float(fq.dtypes)]:
        #     fq.loc[is_zero(fq[col]), col] = 0

        # non-negative variables with a negative value -> set to nan.
        for var in ['be']:
            fq.loc[fq[var] < 0, var] = np.nan  # should be >= 0.

        self.data = fq
        elapsed_time('fundq variables updated.')

    def postprocess(self):
        """Postprocess data.

        This method deletes temporary variable columns (columns starting with '\\_') and replaces infinite
        values with nan.
        """

        elapsed_time('Postprocessing fundq...')
        fq = self.data

        # drop temporary columns that start with '_'
        del_cols = [col for col in fq if col[0] == '_']
        drop_columns(fq, del_cols)

        char_cols = list(self.get_char_list())
        char_cols = list(fq.columns.intersection(char_cols))
        self.inspect_data(char_cols, option=['summary', 'nans'])

        log('Replacing inf with nan...')
        for col in char_cols:
            isinf = (fq[col] == np.inf) | (fq[col] == -np.inf)
            if isinf.any():
                fq.loc[isinf, col] = np.nan

        self.inspect_data(char_cols, option=['stats'])
        elapsed_time('fundq postprocessed.')

    def generate_funda_vars(self):
        """Generate quarterly-updated annual data from fundq.

        The following variables are annualized by cumulating over the past 4 quarters.

            'cogs', 'xsga', 'xint', 'dp', 'txt', 'xrd', 'spi', 'sale', 'revt',
            'xopr', 'oibdp', 'oiadp', 'ib', 'ni', 'xido', 'nopi', 'mii', 'pi', 'xi',
            'oancf', 'dv', 'sstk', 'dlcch', 'capx', 'dltr', 'txbcof', 'xidoc', 'dpc',
            'fiao', 'ibc', 'prstkc', 'fincf'.

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
        tmp = self.rolling(fa[cum_columns].to_numpy(), 4, 'sum').T
        for i, col in enumerate(cum_columns):
            fa.loc[:, col] = tmp[i]

        # For the 4th fiscal quarter, use ytd data instead of the sum of quarterized values. Otherwise, data within the
        # first four quarters will be lost.
        ycolumns = [col for col in self._get_ytd_columns() if
                    col[:-1] in cum_columns]  # columns for ytd data (un-quarterized)
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

        # EDIT: 2023.07.13. In the previous version, dibq was set to nan in the first 4 quarters. This made
        # some valid values in the first 12 quarters become nan. In the new version, we set all nan values of dibq to 0
        # before calculation. After the characteristic is calculated, it is set to nan if dibq was nan.

        fq = self.data
        dibq = self.diff(fq.ibq, 4)
        ni_inc = (dibq > 0).astype(config.float_type)
        char = ni_inc
        for i in range(1, 8):
            idx = ni_inc == i  # True if earnings have increased so far.
            char[idx] += self.shift(ni_inc, i)[idx]

        char[dibq.isna()] = np.nan

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
        if config.replicate_jkp:
            rev_diff = self.diff(fq.saleq, 4)
        else:
            # OP doesn't appear to adjust for split, but we adjust for split following HXZ. (from ver 1.0)
            rev_diff = self.diff(fq.saleq / (fq.cshprq * fq.ajexq), 4)
        rev_diff_mean = self.rolling(rev_diff, 8, 'mean', min_n=6, lag=1)
        rev_diff_std = self.rolling(rev_diff, 8, 'std', min_n=6, lag=1)
        char = (rev_diff - rev_diff_mean) / rev_diff_std
        char[(rev_diff_std < ZERO) | (fq.cshprq < ZERO)] = np.nan

        return char

        # HXZ
        # rev_diff = self.diff(fq.saleq / (fq.cshprq * fq.ajexq), 4)
        # char = rev_diff / self.rolling(rev_diff, 8, 'std', min_n=6)
        # return char

    def c_niq_su(self):
        """Earnings surprise. Foster, Olsen, and Shevlin (1984)"""

        fq = self.data
        if config.replicate_jkp:
            ear_diff = self.diff(fq.ibq, 4)
        else:
            # OP doesn't appear to adjust for split, but we adjust for split following HXZ. (from ver 1.0)
            ear_diff = self.diff(fq.epspxq / fq.ajexq, 4)
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
        if config.replicate_jkp:
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
        char[fq.prccq * fq.cshoq < ZERO] = np.nan
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
class CRSPM(FCPanel):
    """Class to generate firm characteristics from crspm.

    The firm characteristics defined in this class can be viewed using :meth:`.show_available_chars`.

    Args:
        alias (str, list, or dict): Firm characteristics to generate and their aliases. If None, all available firm
            characteristics are generated.

            * str: A column name in the mapping file (``config.mapping_file_path``). The firm characteristics defined
              in this class and in the `alias` column of the mapping file are generated. See :ref:`sec-mapping file`.
            * list: List of firm characteristics (method names).
            * dict: Dict of firm characteristics and their aliases of the form {method name: alias}.

            If aliases are not specified, method names are used as aliases.
        data: DataFrame of crspm data with index = date/permno, sorted on permno/date. The crspm data can be given at
            initialization or loaded later using :meth:`load_data`.

    **Methods**

    .. autosummary::
        :nosignatures:

        load_data
        filter_data
        update_variables
        merge_with_factors
        postprocess

    Firm characteristic generation methods have a name like ``c_characteristic()``.
    """

    def __init__(self, alias=None, data=None):
        super().__init__(alias, data, MONTHLY)

    def load_data(self, sdate=None, edate=None, fname='crspm'):
        """Load crspm data from file.

        This method loads crspm data from ``config.input_dir/crspm`` and stores it in the ``data`` attribute.
        The ``data`` has index = date/permno and is sorted on permno/date.

        NOTE:
            In CRSP monthly tables, date is the last business day of the month, whereas datadate in Compustat is the
            end-of-month date. To make the two dates consistent, crspm dates are shifted to the end of the month.

        Args:
            sdate: Start date ('yyyy-mm-dd'). If None, from the earliest.
            edate: End date ('yyyy-mm-dd'). If None, to the latest.
            fname: The crspm file name.
        """

        elapsed_time(f'Loading {fname}...')
        data = WRDS.read_data(fname, index_col=['date', 'permno'])
        if not self.is_sorted(data):
            data.sort_index(level=[-1, 0], inplace=True)

        if sdate:
            data = data[data.index.get_level_values(0) >= sdate]
        if edate:
            data = data[data.index.get_level_values(0) <= edate]

        data.reset_index(inplace=True)
        data['date'] = to_month_end(data['date'])  # Push date to month end ignoring holidays. compustat datadate is
                                                   # always month end, while crsp date is business day month end.
        data.set_index(['date', 'permno'], inplace=True)

        self.data = data
        elapsed_time(f'{fname} loaded.')
        self.inspect_data()

    # def cleanse_data(self, method='Quantile', **kwargs):
    #     data = self.data.copy()
    #
    #     # Columns to examine.
    #     excl_columns = ['exchcd', 'shrcd', 'permco', 'issuno', 'hexcd', 'hsiccd', 'siccd']
    #     columns = [col for col in data.columns if is_numeric(data[col]) and (col not in excl_columns)]
    #     keep_columns(data, columns)
    #
    #     # Adjust for splits.
    #     data.loc[data.cfacpr == 0, 'cfacpr'] = np.nan
    #     data.loc[data.cfacshr == 0, 'cfacshr'] = np.nan
    #
    #     data['prc'] = data.prc.abs() / data.cfacpr
    #     data['altprc'] = data.altprc.abs() / data.cfacpr
    #     data['bidlo'] = data.bidlo.abs() / data.cfacpr
    #     data['askhi'] = data.askhi.abs() / data.cfacpr
    #     data['bid'] /= data.cfacpr
    #     data['ask'] /= data.cfacpr
    #     data['shrout'] *= data.cfacshr
    #
    #     anomaly = detect_anomalies(data, None, method, **kwargs)
    #     for i, col in enumerate(columns):
    #         self.data.loc[anomaly[:, i], col] = np.nan

    def filter_data(self):
        """Filter data.

        The data is filtered using the following filters:

            * shrcd in [10, 11, 12]

        Note:
            We do not filter the data using exchange code (exchcd in [1 (NYSE), 2 (ASE), 3 (NASDAQ)]) because exchcd
            can change when a stock is delisted: If the data is filtered using exchcd, the data of the delist month can
            be lost.
        """

        elapsed_time('Filtering crspm...')
        log(f'crspm shape before filtering: {self.data.shape}')

        # Ordinary Common Shares: shrcd in (10, 11, 12).
        self.filter(('shrcd', 'in', [10, 11, 12]))

        # Exchange filtering: 1 (NYSE), 2 (ASE), 3 (NASDAQ)
        # We prefer to not filter on exchcd as it can change in the month when a stock is delisted.
        # self.filter(('exchcd', 'in', [1, 2, 3]))

        log(f'crspm shape after filtering: {self.data.shape}')
        elapsed_time('crspm filtered.')

    def update_variables(self):
        """Preprocess data before creating firm characteristics.

        * Convert negative prices (quotes) to positive.
        * Convert shares outstanding (shrout) unit from thousands to millions (divide by 1000).
        * Convert trading volume (vol) unit from 100 shares to shares (multiply by 100).
        * Adjust trading volume following Gao and Ritter (2010).
        * Create frequently used variables.
        """

        elapsed_time('Updating crspm variables...')

        self.data.reset_index(inplace=True)
        cm = self.data

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
        cm['me_company'] = cm.groupby(['date', 'permco'])['me'].transform('sum')  # firm-level market equity
        cm.loc[cm.me.isna(), 'me_company'] = np.nan

        # Risk-free rate and excess return
        rf = WRDS.get_risk_free_rate(month_end=True)
        cm = merge(cm, rf, on='date', how='left')
        cm.set_index(['date', 'permno'], inplace=True)

        cm['exret'] = cm.ret - cm.rf
        if config.replicate_jkp:
            cm['exret'] = winsorize(cm['exret'], (1e-3, 1e-3))  # winsorize at 0.1%, 99.9%

        self.data = cm
        elapsed_time('crspm variables updated.')

    def merge_with_factors(self, factors=None):
        """Merge crspm with factors.

        The `factors` should contain Fama-French 3 factors with column names as defined in ``config.factor_names``.

        Args:
            factors: DataFrame of factors with index = date or list of factors. If None or list, factor data will be
                read from ``config.monthly_factors_fname`` and only the factors in `factors` will be merged.
        """

        elapsed_time('Merging factors with crspm...')
        if factors is None:
            factors = read_from_file(config.factors_monthly_fname)
        elif isinstance(factors, list):
            factors = read_from_file(config.factors_monthly_fname)[factors]


        # Rename factor names as the keys of config.factor_names.
        factors = factors.rename(columns={v: k for k, v in config.factor_names.items()})
        if 'rf' in factors:  # drop rf as it is already in crspm.
            del factors['rf']

        factors.index = to_month_end(factors.index)
        self.merge(factors, on='date', how='left')
        elapsed_time('Factors merged with crspm.')

    def postprocess(self):
        """Postprocess data.

        This method deletes temporary variable columns (columns starting with '\\_') and replaces infinite
        values with nan.
        """

        elapsed_time('Postprocessing crspm...')
        cm = self.data

        # drop temporary columns that start with '_'
        del_cols = [col for col in cm if col[0] == '_']
        drop_columns(cm, del_cols)

        char_cols = list(self.get_char_list())
        char_cols = list(cm.columns.intersection(char_cols))
        self.inspect_data(char_cols, option=['summary', 'nans'])

        log('Replacing inf with nan...')
        for col in char_cols:
            isinf = (cm[col] == np.inf) | (cm[col] == -np.inf)
            if isinf.any():
                cm.loc[isinf, col] = np.nan

        self.inspect_data(char_cols, option=['stats'])
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
        div = (cm.ret - cm.retx) * cm.cfacpr * self.shift(cm.prc / cm.cfacpr) * cm.shrout
        div12m = self.rolling(div, 12, 'sum')  # Total dividend over the past 1 year.
        # Set char = nan if div_tot = 0 (JKP do not impose this). (from ver 1.0)
        char = np.where(is_zero(div12m) | is_zero(cm.me), np.nan, div12m / cm.me)
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
        if config.replicate_jkp:
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
        cm = merge(cm, indmom, on=['date', 'sic2'], how='left')

        return cm['indmom']

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

        res = self.rolling_regression(self.data.loc[:, ['exret', 'mktrf']], 60, 36)
        return res[:, 1]

    def c_price(self):
        """Share price. Miller and Scholes (1982)"""

        return self.data['prc']

    @staticmethod
    @njit(cache=True)
    def _residual_momentum(data, window, minobs):
        nobs = data.shape[0]
        retval = np.full((nobs, 2), np.nan, dtype=data.dtype)

        if nobs < minobs:
            return retval

        not_nan = ~isnan1(data)

        Y = data[:, 0]
        X = add_constant(data[:, 1:])

        for i in range(minobs - 1, nobs):
            s, e = i - min(i, window - 1), i + 1

            not_na = not_nan[s:e]
            y = Y[s:e][not_na]
            x = X[s:e][not_na]

            if y.shape[0] < minobs:
                continue

            try:
                _, _, res = regression(y, x)
            except:
                continue

            retval[i, 0] = res[-6:-1].mean() / nanstd(res[-6:-1])
            retval[i, 1] = res[-12:-1].mean() / nanstd(res[-12:-1])

        return retval

    def c_resff3_6_1(self):
        """6 month residual momentum. Blitz, Huij, and Martens (2011)"""

        data = self.data.loc[:, ['exret', 'mktrf', 'hml', 'smb_ff']]
        res = self.apply_to_ids(data, self._residual_momentum, 2, 36, 24)
        self.data['resff3_6_1'] = res[:, 0]
        self.data['resff3_12_1'] = res[:, 1]

        return self.data['resff3_6_1']

    def c_resff3_12_1(self):
        """12 month residual momentum. Blitz, Huij, and Martens (2011)"""

        data = self.data.loc[:, ['exret', 'mktrf', 'hml', 'smb_ff']]
        res = self.apply_to_ids(data, self._residual_momentum, 2, 36, 24)
        self.data['resff3_6_1'] = res[:, 0]
        self.data['resff3_12_1'] = res[:, 1]

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

        # We follow CZ: use 6 months rather than 12 months of OP. (from ver 1.0)
        cm = self.data
        div = cm.cash_div.fillna(0)
        divinit = (div > 0) & (self.rolling(div, 24, 'sum', lag=1) == 0)
        char = self.rolling(divinit, 6, 'sum') > 0
        return char.astype(config.float_type)

        # This is GHZ version
        # div = cm.cash_div.fillna(0)
        # cum_div = self.shift(div).groupby('permno').cumsum()  # sum of previous dividends
        # char = (cum_div == 0) & (div > 0) & (cm.get_row_count() >= 24)
        # return char.astype(config.float_type)

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

        # Omission of quarterly dividend
        char1 = (div == 0) & (div3 > 0) & (div6 > 0) & (div9 > 0) & (div12 > 0) & (div15 > 0) & (div18 > 0)
        # Omission of semi-annual dividend
        char2 = (div == 0) & (div3 == 0) & (div6 > 0) & (div9 == 0) & (div12 > 0) & (div15 == 0) & (div18 > 0)
        # Omission of annual dividend.
        char3 = (div == 0) & (div3 == 0) & (div6 == 0) & (div9 == 0) & (div12 > 0) & (div15 == 0) & (div18 == 0) & (
                    div24 > 0)

        divomit = (char1 | char2 | char3)
        char = self.rolling(divomit, 2, 'sum') > 0  # (from ver 1.0)
        return char.astype(config.float_type)

    def c_dolvol(self):
        """Dollar trading volume (Org, GHZ). Brennan, Chordia, and Subrahmanyam (1998)"""

        # GHZ implementation (Original definition)
        cm = self.data
        dolvol = cm.vol * cm.prc
        dolvol[dolvol < ZERO] = np.nan
        char = self.shift(np.log(dolvol))
        return char

    def c_turn(self):
        """Share turnover (Org, GHZ). Datar, Naik, and Radcliffe (1998)"""

        cm = self.data
        char = self.rolling(cm.vol, 3, 'mean') / (cm.shrout * 1000)  # shrout is divided by 1000 in update_variables().
        return char

    def c_ipo(self):
        """Initial public offerings. Loughran and Ritter (1995)"""

        cm = self.data
        return (self.get_row_count() < 12).astype(config.float_type)


################################################################################
#
# CRSPD
#
################################################################################
class CRSPDRaw(FCPanel):
    """Class that handles crspd data.

    This class contains daily crspd data and is used to generate monthly firm characteristics in :class:`CRSPD`.

    Args:
        alias (str, list, or dict): Firm characteristics to generate and their aliases. If None, all available firm
            characteristics are generated.

            * str: A column name in the mapping file (``config.mapping_file_path``). The firm characteristics defined
              in this class and in the `alias` column of the mapping file are generated. See :ref:`sec-mapping file`.
            * list: List of firm characteristics (method names).
            * dict: Dict of firm characteristics and their aliases of the form {method name: alias}.

            If aliases are not specified, method names are used as aliases.
        data: DataFrame of crspd data with index = date/permno, sorted on permno/date. The crspd data can be given at
            initialization or loaded later using :meth:`load_data`.

    **Methods**

    .. autosummary::
        :nosignatures:

        load_data
        filter_data
        update_variables
        merge_with_factors
        get_idym_group
        get_idym_group_size
        apply_to_idyms
    """

    def __init__(self, alias=None, data=None):
        super().__init__(alias, data, DAILY)

        self._gb_idym_size = None

    def load_data(self, sdate=None, edate=None, fname='crspd'):
        """Load crspd data from file.

        This method loads crspd data from ``config.input_dir/crspd`` and stores it in the ``data`` attribute.
        The ``data`` has index = date/permno and is sorted on permno/date.

        Args:
            sdate: Start date ('yyyy-mm-dd'). If None, from the earliest.
            edate: End date ('yyyy-mm-dd'). If None, to the latest.
            fname: The crspd file name.
        """

        elapsed_time(f'Loading {fname}...')
        data = WRDS.read_data(fname, index_col=['date', 'permno'])
        if not self.is_sorted(data):
            data.sort_index(level=[-1, 0], inplace=True)

        if sdate:
            data = data[data.index.get_level_values(0) >= sdate]
        if edate:
            data = data[data.index.get_level_values(0) <= edate]

        self.data = data
        elapsed_time(f'{fname} loaded.')
        self.inspect_data()

    # def cleanse_data(self, method='Quantile', **kwargs):
    #     data = self.data.copy()
    #
    #     # Columns to examine.
    #     excl_columns = ['exchcd', 'shrcd', 'permco']
    #     columns = [col for col in data.columns if is_numeric(data[col]) and (col not in excl_columns)]
    #     keep_columns(data, columns)
    #
    #     # Adjust for splits.
    #     data.replace({'cfacpr': 0, 'cfacshr': 0}, np.nan, inplace=True)
    #     # data.loc[data.cfacpr == 0, 'cfacpr'] = np.nan
    #     # data.loc[data.cfacshr == 0, 'cfacshr'] = np.nan
    #
    #     data['prc'] = data.prc.abs() / data.cfacpr
    #     data['bidlo'] = data.bidlo.abs() / data.cfacpr
    #     data['askhi'] = data.askhi.abs() / data.cfacpr
    #     data['shrout'] *= data.cfacshr
    #
    #     anomaly = detect_anomalies(data, None, method, **kwargs)
    #     for i, col in enumerate(columns):
    #         self.data.loc[anomaly[:, i], col] = np.nan

    def filter_data(self):
        """Filter data.

        The data is filtered using the following filters:

            * shrcd in [10, 11, 12]

        Note:
            We do not filter the data using exchange code (exchcd in [1 (NYSE), 2 (ASE), 3 (NASDAQ)]) because exchcd
            can change when a stock is delisted: If the data is filtered using exchcd, the data of the delist month can
            be lost.
        """

        elapsed_time('Filtering crspd...')
        log(f'crspd shape before filtering: {self.data.shape}')

        # Ordinary Common Shares: shrcd in (10, 11, 12).
        self.filter(('shrcd', 'in', [10, 11, 12]))

        # Exchange filtering: 1 (NYSE), 2 (ASE), 3 (NASDAQ)
        # We prefer to not filter on exchcd as it can change in the month when a stock is delisted.
        # self.filter(('exchcd', 'in', [1, 2, 3]))

        log(f'crspd shape after filtering: {self.data.shape}')
        elapsed_time('crspd filtered.')

    def update_variables(self):
        """Preprocess data before creating firm characteristics.

        * Convert negative prices (quotes) to positive.
        * Set askhi and bidlo to nan if it is negative, the price is negative, or the volume is 0.
        * Conver cfacpr of 0 to nan.
        * Convert shares outstanding (shrout) unit from thousands to millions (divide by 1000).
        * Adjust trading volume following Gao and Ritter (2010).
        * Create frequently used variables.
        """

        elapsed_time('Updating crspd variables...')
        self.data.reset_index(inplace=True)
        cd = self.data

        cd.loc[(cd.askhi <= 0) | (cd.prc <= 0) | (cd.vol == 0), 'askhi'] = np.nan
        cd.loc[(cd.bidlo <= 0) | (cd.prc <= 0) | (cd.vol == 0), 'bidlo'] = np.nan
        cd.loc[:, 'prc'] = cd.prc.abs().to_numpy()

        if config.replicate_jkp:
            cd.replace({'cfacshr': 0}, np.nan, inplace=True)  # JKP use cfacshr instead of cfacpr
            adj_factor = cd.cfacshr
        else:
            cd.replace({'cfacpr': 0}, np.nan, inplace=True)
            adj_factor = cd.cfacpr

        cd['prc_adj'] = cd.prc / adj_factor
        cd['bidlo_adj'] = cd.bidlo / adj_factor
        cd['askhi_adj'] = cd.askhi / adj_factor

        cd.loc[:, 'shrout'] /= 1000  # to millions

        # Adjust trading volume following Gao and Ritter (2010)
        date = cd['date']
        vol = cd.vol.to_numpy()
        cd.loc[:, 'vol'] = np.where((cd.exchcd == 3) & (date <= '2001-01-31'), vol / 2, vol)
        cd.loc[:, 'vol'] = np.where((cd.exchcd == 3) & (date > '2001-01-31') & (date <= '2001-12-31'), vol / 1.8, vol)
        cd.loc[:, 'vol'] = np.where((cd.exchcd == 3) & (date > '2001-12-31') & (date <= '2003-12-31'), vol / 1.6, vol)
        # cd.loc[(cd.exchcd == 3) & (date <= '2001-01-31'), 'vol'] /= 2
        # cd.loc[(cd.exchcd == 3) & (date > '2001-01-31') & (date <= '2001-12-31'), 'vol'] /= 1.8
        # cd.loc[(cd.exchcd == 3) & (date > '2001-12-31') & (date <= '2003-12-31'), 'vol'] /= 1.6

        cd['me'] = cd.prc * cd.shrout
        cd['me_company'] = cd.groupby(['date', 'permco'])['me'].transform('sum')  # firm-level market equity
        cd['dvol'] = cd.prc * cd.vol

        cd['mon'] = (date.dt.year - 1900) * 12 + date.dt.month
        cd['ym'] = to_month_end(date)

        # Risk-free rate and excess return
        rf = WRDS.get_risk_free_rate(month_end=True) / 21
        cd = merge(cd, rf, on='ym', right_on='date', how='left')
        cd.set_index(['date', 'permno'], inplace=True)

        cd['exret'] = cd.ret - cd.rf
        if config.replicate_jkp:
            # JKP winsorise excess return only in crspm. We do so also in crspd for consistency.
            cd['exret'] = winsorize(cd['exret'], (1e-3, 1e-3))  # winsorize at 0.1%, 99.9% (slower)
        cd['turnover_d'] = (cd.vol / (cd.shrout * 1e6)).astype(cd.vol.dtype)

        # If number of zero returns >= 10 in a month, all returns of the month are set to nan.
        cd['_tmp'] = cd['ret'] == 0
        zero_ret_cnt = self.get_idym_group()._tmp.transform('sum')
        cd.loc[zero_ret_cnt >= 10, 'exret'] = np.nan

        elapsed_time('crspd variables updated.')

    def merge_with_factors(self, factors=None):
        """Merge crspd with factors.

        The `factors` should contain Fama-French 3 factors and Hou-Xue-Zhang 4 factors with column names as defined
        in ``config.factor_names``.

        Args:
            factors: DataFrame of factors with index = date or list of factors. If None or list, factor data will be
                read from ``config.daily_factors_fname`` and only the factors in `factors` will be merged.
        """

        elapsed_time('Merging factors with crspd...')
        if factors is None:
            factors = read_from_file(config.factors_daily_fname)
        elif isinstance(factors, list):
            factors = read_from_file(config.factors_daily_fname)[factors]

        # Rename factor names as the keys of config.factor_names.
        factors.rename(columns={v: k for k, v in config.factor_names.items()}, inplace=True)
        if 'rf' in factors:  # drop rf if it is already in crspm.
            del factors['rf']

        self.merge(factors, on='date', how='left')
        elapsed_time('Factors merged with crspd.')

    def get_idym_group(self):
        """Get id-month group.

        Group ``data`` by permno and year-month and return the GroupBy object.

        Returns:
            Pandas GroupBy object.
        """

        return self.data.groupby(['permno', 'ym'])

    def get_idym_group_size(self):
        """Get id-month group sizes.

        Returns:
            Ndarray of id-month group sizes.
        """

        if (self._gb_idym_size is None) or (np.sum(self._gb_idym_size) != self.data.shape[0]):
            _gb_idym = self.get_idym_group()
            self._gb_idym_size = _gb_idym.size().to_numpy()

        return self._gb_idym_size

    def apply_to_idyms(self, data, function, n_ret, *args, data2=None):
        """Apply a function to each id-month group.

        This method groups `data` by id-month and applies `function` to each group.
        This method can be used when the `function` is a reduce function and requires only the data within the month.

        Args:
            data: DataFrame, Series, ndarray, or (list of) columns. It should have the same length and order as the
                  ``data`` attribute.
            function: Jitted reduce function to apply to groups. Its arguments should be (`gbdata`, `*args`) or
                (`gbdata`, `gbdata2`, `*args`), where `gbdata` (`gbdata2`) is a group of `data` (`data2`).
            n_ret: Number of returns of `function`. If None, it is assumed to be the same as the column size of `data`.
            *args: Additional arguments of `function`.
            data2: DataFrame, Series, ndarray, str, or int. Optional argument when `function` requires two sets of
                input data.

        Returns:
            Concatenated value of the outputs of `function`. Size = (number of id-month groups) x `n_retval`.

        See Also:
            :func:`~.datatools.apply_to_groups_reduce_jit`
        """

        value = self._to_value(data)
        value2 = self._to_value(data2) if data2 is not None else None
        gsize = self.get_idym_group_size()

        return apply_to_groups_reduce_jit(value, gsize, function, n_ret, *args, data2=value2)


################################################################################
# Local functions used in CRSPD
# - These functions are defined outside the CRSPD class so that they can be
#   cached. If defined inside as a staticmethod, they cannot be called in other
#   cached staticmethods.
################################################################################

# Functions used in CRSPD._apply_to_ids

@njit(error_model='numpy', cache=True)
def crspd_corr_1260d(data):
    unique_mons = np.unique(data[:, 0])
    data = data[~isnan1(data[:, 1:])]
    mons = data[:, 0]
    data = data[:, 1:]

    corr = np.full((unique_mons.shape[0], 1), np.nan)
    for j in range(unique_mons.shape[0]):
        mon = unique_mons[j]
        data_m = data[(mon - 60 < mons) & (mons <= mon)]
        if data_m.shape[0] >= 750:
            corr[j] = np.corrcoef(data_m.T)[0, 1]

    return corr

@njit(error_model='numpy', cache=True)
def crspd_roll_126d(data):
    mons = data[:, 0]
    data = data[:, 1:]
    unique_mons = np.unique(mons)

    res = np.full((unique_mons.shape[0], 6), np.nan, dtype=data.dtype)
    ami = res[:, 0]
    zero = res[:, 1]
    dolvol = res[:, 2]
    dolvol_var = res[:, 3]
    turnover = res[:, 4]
    turnover_var = res[:, 5]

    for j in range(unique_mons.shape[0]):
        mon = unique_mons[j]
        data_m = data[(mon - 6 < mons) & (mons <= mon)]
        isnan = isnan1(data_m[:, :1])
        ret = data_m[~isnan, 0]
        dvol = data_m[~isnan, 1]
        if (dvol.shape[0] >= 60) & (dvol != 0).any():  # HXZ require minimum 50 obs, while JKP require 60.
            ami[j] = 1e6 * (np.abs(ret[dvol != 0]) / dvol[dvol != 0]).mean()
            dolvol[j] = dvol.mean()
            dolvol_var[j] = nanstd(dvol) / dolvol[j]

        vol = data_m[:, 2]
        turnover_d = data_m[:, 3]
        turnover_d = turnover_d[~np.isnan(turnover_d)]
        if turnover_d.shape[0] >= 60:
            turnover_sum = turnover_d.sum()
            if turnover_sum == 0:  # EDIT: 2023.07.13. Omit 0 turnover to prevent inf.
                zero[j] = (vol == 0).sum() * 126 / turnover_d.shape[0]
            else:
                zero[j] = ((vol == 0).sum() + 1 / turnover_sum / 11000) * 126 / turnover_d.shape[0]
            turnover[j] = turnover_sum / turnover_d.shape[0]
            turnover_var[j] = nanstd(turnover_d) / turnover[j]

    return res

@njit(error_model='numpy', cache=True)
def crspd_roll_252d(data):
    mons = data[:, 0]
    unique_mons = np.unique(mons)

    res = np.full((unique_mons.shape[0], 6), np.nan, dtype=data.dtype)
    rvol = res[:, 0]
    mktvol = res[:, 1]
    ivol = res[:, 2]
    prc_high = res[:, 3]
    zero = res[:, 4]
    betadown = res[:, 5]

    for j in range(unique_mons.shape[0]):
        mon = unique_mons[j]
        data_m = data[(mon - 12 < mons) & (mons <= mon), 1:]

        # rvol, mktvol, ivol
        retdata = data_m[:, :2]  # exret, mktrf
        retdata = retdata[~isnan1(retdata)]
        if retdata.shape[0] >= 120:
            endo = retdata[:, 0]
            exog = retdata[:, 1]

            rvol[j] = nanstd(endo)
            mktvol[j] = nanstd(exog)

            cov = np.cov(endo, exog)
            beta = cov[0][1] / cov[1][1]
            c = endo.mean() - beta * exog.mean()
            pred = c + beta * exog
            ivol[j] = nanstd(endo - pred)

        # 52-week high
        prc_adj = data_m[:, 2]
        if np.sum(~np.isnan(prc_adj)) >= 120:
            prc_high[j] = prc_adj[-1] / np.nanmax(prc_adj)

        # zero-trades
        vol = data_m[:, 3]
        turnover_d = data_m[:, 4]
        turnover_d = turnover_d[~np.isnan(turnover_d)]
        if turnover_d.shape[0] >= 120:
            turnover_sum = turnover_d.sum()
            if turnover_sum == 0:  # EDIT: 2023.07.13. Omit 0 turnover to prevent inf.
                zero[j] = (vol == 0).sum() * 252 / turnover_d.shape[0]
            else:
                zero[j] = ((vol == 0).sum() + 1 / turnover_sum / 11000) * 252 / turnover_d.shape[0]

        # betadown
        downdata = retdata[retdata[:, 1] < 0]  # mktrf < 0
        if downdata.shape[0] >= 60:
            cov = np.cov(downdata.T)
            betadown[j] = cov[0][1] / cov[1][1]

    return res

# Functions used in CRSPD._apply_to_idyms

@njit(cache=True)
def crspd_rmax5_21d(x):
    if x.shape[0] <= 5:
        return np.nan
    else:
        return -np.mean(np.partition(-x, 5)[:5])


@njit(cache=True)
def crspd_rskew_21d(data):
    data = data[~isnan1(data)]
    if data.shape[0] >= 3:
        return nanskew(data)
    else:
        return np.nan


@njit(cache=True)
def crspd_icapm_21d(data):
    data = data[~isnan1(data)]
    if data.shape[0] < 15:
        return np.nan, np.nan, np.nan

    y = data[:, 0]
    x = data[:, 1]

    beta, r2, res = bivariate_regression(y, x)

    ivol = nanstd(res)
    iskew = nanskew(res)
    x_dm = x - x.mean()
    coskew = (res * x_dm ** 2).mean() / ((res ** 2).mean() ** 0.5 * (x_dm ** 2).mean())

    return ivol, iskew, coskew


@njit(cache=True)
def crspd_multifactor_ivol_skew(data):
    """Calculate idiosyncratic volatility and skewness from a multi-factor model.

    The factor regression is run every month for each stock using the sample within the month.

    Args:
        reg_vars: List of regressand (excess return) and regressors (factors). Regressand should be the first
            element of `reg_vars`.

    Returns:
        Ivol and skew. N x 2 ndarray, where N is the number of permon-month pairs.

    Examples:
        Assuming that `cd` is an instance of ``CRSPDRaw``, ivol and skew from Fama-French three-factor model can be
        obtained as follows.

        >>> reg_vars = ['exret', 'mktrf', 'hml', 'smb_ff']  # y, X
        ... cd.multifactor_ivol_skew(reg_vars)
    """

    data = data[~isnan1(data)]
    if data.shape[0] < 15:
        return np.nan, np.nan

    y = data[:, 0].copy()
    X = add_constant(data[:, 1:])
    _, _, res = regression(y, X)

    # if config.replicate_jkp:
    #     return nanstd(res) * np.sqrt((data.shape[0] - 1) / (data.shape[0] - data.shape[1] + 1)), nanskew(res)
    # else:
    return nanstd(res), nanskew(res)


@njit(cache=True)
def crspd_beta_dimson(data):
    data = data[:-1]  # remove the last row that contains a future market return (mktrf_ld1)
    data = data[~isnan1(data)]
    if data.shape[0] < 15:
        return np.nan

    endo = data[:, 0].copy()
    exog = add_constant(data[:, 1:])

    beta, _, _ = regression(endo, exog)
    dimson = beta[1:].sum()

    return dimson


class CRSPD(FCPanel):
    """Class to generate firm characteristics from crspd.

    This class has a ``CRSPDRaw`` object as a member attribute and use it to generate monthly firm characteristics.
    ``CRSPDRaw.data`` contains daily crspd data and ``CRSPD.data`` contains monthly firm characteristics.
    The firm characteristics defined in this class can be viewed using :meth:`.show_available_chars`.

    Args:
        alias (str, list, or dict): Firm characteristics to generate and their aliases. If None, all available firm
            characteristics are generated.

            * str: A column name in the mapping file (``config.mapping_file_path``). The firm characteristics defined
              in this class and in the `alias` column of the mapping file are generated. See :ref:`sec-mapping file`.
            * list: List of firm characteristics (method names).
            * dict: Dict of firm characteristics and their aliases of the form {method name: alias}.

            If aliases are not specified, method names are used as aliases.
        data: DataFrame of crspd data with index = date/permno, sorted on permno/date. The crspd data can be given at
            initialization or loaded later using :py:meth:`load_data`.

    **Attributes**

    Attributes:
        cd: :class:`CRSPDRaw` object that stores daily crspd data.

    **Methods**

    .. autosummary::
        :nosignatures:

        load_data
        filter_data
        update_variables
        merge_with_factors
        postprocess
        get_idym_group
        get_idym_group_size

    Firm characteristic generation methods have a name like ``c_characteristic()``.
    """

    def __init__(self, alias=None, data=None):
        super().__init__(alias, None, MONTHLY)  # Characteristics are generated monthly.
        self.cd = CRSPDRaw(alias, data)

    def load_data(self, sdate=None, edate=None, fname='crspd'):
        """Load crspd data from file.

        This is a wrapping method of :meth:`CRSPDRaw.load_data`.
        """

        self.cd.load_data(sdate, edate, fname)

    def filter_data(self):
        """Filter data.

        This is a wrapping method of :meth:`CRSPDRaw.filter_data`.
        """

        self.cd.filter_data()

    def update_variables(self):
        """Preprocess data before creating firm characteristics.

        This method calls :meth:`CRSPDRaw.update_variables` to update variables and initializes the ``data`` attribute.
        """

        self.cd.update_variables()

        # Initialize firm characteristics data.
        self.data = pd.DataFrame(index=self.get_idym_group()['ym'].last().index, dtype=config.float_type).swaplevel()
        self.data.index.names = ('date', 'permno')

    def merge_with_factors(self, factors=None):
        """Merge crspd with factors.

        This is a wrapping method of :py:meth:`CRSPDRaw.merge_with_factors`.
        """

        self.cd.merge_with_factors(factors)

    def postprocess(self):
        """Postprocess data.

        This method deletes temporary variable columns (columns starting with '\\_') and replaces infinite
        values with nan.
        """

        elapsed_time('Postprocessing crspd...')
        cd = self.data

        # drop temporary columns that start with '_'
        del_cols = [col for col in cd if col[0] == '_']
        drop_columns(cd, del_cols)

        char_cols = list(self.get_char_list())
        char_cols = list(cd.columns.intersection(char_cols))
        self.inspect_data(char_cols, option=['summary', 'nans'])

        log('Replacing inf with nan...')
        for col in char_cols:
            isinf = (cd[col] == np.inf) | (cd[col] == -np.inf)
            if isinf.any():
                cd.loc[isinf:, col] = np.nan

        self.inspect_data(char_cols, option=['stats'])
        elapsed_time('crspd postprocessed.')

    def get_idym_group(self):
        """Get id-month group.

        This is a wrapping method of :meth:`CRSPDRaw.get_idym_group`.
        """

        return self.cd.get_idym_group()

    def get_idym_group_size(self):
        """Get id-month group sizes.

        This is a wrapping method of :meth:`CRSPDRaw.get_idym_group_size`.
        """

        return self.cd.get_idym_group_size()

    #######################################
    # Characteristics
    #######################################

    def _apply_to_ids(self, data, function, n_ret, *args):

        @njit(parallel=True, cache=True)
        def inner(x, gsize, gsize2, n_ret, *args):

            idx = np.full(1 + gsize.shape[0], 0)
            idx[1:] = np.cumsum(gsize)
            idx2 = np.full(1 + gsize2.shape[0], 0)
            idx2[1:] = np.cumsum(gsize2)

            retval = np.full((idx2[-1], n_ret), np.nan, dtype=x.dtype)
            if function == 'corr_1260d':
                fcn = crspd_corr_1260d
            elif function == 'roll_126d':
                fcn = crspd_roll_126d
            elif function == 'roll_252d':
                fcn = crspd_roll_252d
            else:
                raise ValueError(f'Unsupported function: {function} ')

            for i in prange(gsize.shape[0]):
                retval[idx2[i]:idx2[i + 1]] = fcn(x[idx[i]:idx[i + 1]], *args)

            return retval

        value = self.cd._to_value(data)
        gsize = self.cd.get_id_group_size()
        gsize2 = self.get_id_group_size()

        retval = inner(value, gsize, gsize2, n_ret, *args)
        del value
        return retval


    def _apply_to_idyms(self, data, function, n_ret, *args):

        @njit(parallel=True, cache=True)
        def inner(x, gsize, n_ret, *args):

            retval = np.full((gsize.shape[0], n_ret), np.nan, dtype=x.dtype)

            idx = np.full(1 + gsize.shape[0], 0)
            idx[1:] = np.cumsum(gsize)

            if function == 'rmax_21d':
                fcn = crspd_rmax5_21d
            elif function == 'rskew_21d':
                fcn = crspd_rskew_21d
            elif function == 'icapm_21d':
                fcn = crspd_icapm_21d
            elif function == 'multifactor_ivol_skew':
                fcn = crspd_multifactor_ivol_skew
            elif function == 'beta_dimson':
                fcn = crspd_beta_dimson
            else:
                raise ValueError(f'Unsupported function: {function} ')

            for i in prange(gsize.shape[0]):
                retval[i] = fcn(x[idx[i]:idx[i + 1]], *args)

            return retval

        value = self.cd._to_value(data)
        gsize = self.get_idym_group_size()

        res = inner(value, gsize, n_ret, *args)
        del value

        return res if n_ret > 1 else res.ravel()

    def c_retvol(self):
        """Return volatility. Ang et al. (2006)"""

        return self.get_idym_group().ret.std().to_numpy()

    def c_rmax1_21d(self):
        """Maximum daily return. Bali, Cakici, and Whitelaw (2011)"""

        return self.get_idym_group().ret.max().to_numpy()

    def c_rmax5_21d(self):
        """Highest 5 days of return. Bali, Brown, and Tang (2017)"""

        # EDIT: 2023.07.13. In the previous logic using pd.rank, if there are 2 positive returns and many 0s,
        # the average of the two positive returns was returned. In the new logic, the average of five returns
        # incl. 0s is returned. Also, if N(non-nan returns) <=5, the result is nan.

        return self._apply_to_idyms('ret', 'rmax_21d', 1)

    def c_rskew_21d(self):
        """Return skewness. Bali, Engle, and Murray (2016)"""

        return self._apply_to_idyms('ret', 'rskew_21d', 1)

    def _icapm_21d(self):
        """ivol, skew, coskew from CAPM"""

        if self.created('ivol_capm_21d'):
            return

        reg_vars = ['exret', 'mktrf']  # y, X
        ocolumns = ['ivol_capm_21d', 'iskew_capm_21d', 'coskew_capm_21d']

        self.data[ocolumns] = self._apply_to_idyms(reg_vars, 'icapm_21d', 3)

    def c_ivol_capm_21d(self):
        """Idiosyncratic volatility (CAPM). Ang et al. (2006)"""

        self._icapm_21d()
        return self.data['ivol_capm_21d']

    def c_iskew_capm_21d(self):
        """Idiosyncratic skewness (CAPM). Bali, Engle, and Murray (2016)"""

        self._icapm_21d()
        return self.data['iskew_capm_21d']

    def c_coskew_21d(self):
        """Coskewness. Harvey and Siddique (2000)"""

        self._icapm_21d()
        return self.data['coskew_capm_21d']

    def _iff3_21d(self):
        if self.created('ivol_ff3_21d'):
            return

        reg_vars = ['exret', 'mktrf', 'hml', 'smb_ff']  # y, X
        ocolumns = ['ivol_ff3_21d', 'iskew_ff3_21d']
        self.data[ocolumns] = self._apply_to_idyms(reg_vars, 'multifactor_ivol_skew', 2)

    def c_ivol_ff3_21d(self):
        """Idiosyncratic volatility (FF3). Ang et al. (2006)"""

        self._iff3_21d()
        return self.data['ivol_ff3_21d']

    def c_iskew_ff3_21d(self):
        """Idiosyncratic skewness (FF3). Bali, Engle, and Murray (2016)"""

        self._iff3_21d()
        return self.data['iskew_ff3_21d']

    def _ihxz4_21d(self):
        if self.created('ivol_hxz4_21d'):
            return

        reg_vars = ['exret', 'mktrf', 'smb_hxz', 'inv', 'roe']  # y, X
        ocolumns = ['ivol_hxz4_21d', 'iskew_hxz4_21d']
        self.data[ocolumns] = self._apply_to_idyms(reg_vars, 'multifactor_ivol_skew', 2)

    def c_ivol_hxz4_21d(self):
        """Idiosyncratic volatility (HXZ). Ang et al. (2006)"""

        self._ihxz4_21d()
        return self.data['ivol_hxz4_21d']

    def c_iskew_hxz4_21d(self):
        """Idiosyncratic skewness (HXZ). Bali, Engle, and Murray (2016)"""

        self._ihxz4_21d()
        return self.data['iskew_hxz4_21d']

    def c_beta_dimson_21d(self):
        """Dimson Beta. Dimson (1979)"""

        reg_data = self.cd[['exret', 'mktrf']]
        reg_data['mktrf_lg1'] = self.cd.shift(reg_data['mktrf'])
        reg_data['mktrf_ld1'] = self.cd.shift(reg_data['mktrf'], -1)

        return self._apply_to_idyms(reg_data, 'beta_dimson', 1)

    def c_zero_trades_21d(self):
        """Zero-trading days (1 month). Liu (2006)"""

        self.cd['_tmp'] = self.cd['vol'] == 0
        gb = self.get_idym_group()

        # EDIT: 2023.07.13. Before, cnt was the number of days in the month. Now, it is the days of valid turnover_d.
        cnt = gb.turnover_d.count()
        zero_days = gb._tmp.sum()
        turnover = gb.turnover_d.sum()

        # EDIT: 2023.07.13. Omit 0 turnover to prevent inf.
        zero_days = (zero_days + np.where(turnover == 0, 0, 1 / turnover / 480000)) * 21 / cnt
        zero_days[cnt == 0] = np.nan

        return zero_days.to_numpy()

    @staticmethod
    @njit(error_model='numpy', cache=True)
    def _adjust_bidaskhl(prc, bidlo, askhi, rcount):
        """Adjust bidlo, askhi."""

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

        cd = self.cd.data

        bidlo, askhi = self._adjust_bidaskhl(cd.prc_adj.to_numpy(), cd.bidlo_adj.to_numpy(), cd.askhi_adj.to_numpy(),
                                             self.cd.get_row_count())

        prc_l1 = self.cd.shift(cd.prc.values)
        lo_l1 = self.cd.shift(bidlo, 1)
        hi_l1 = self.cd.shift(askhi, 1)

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

        cd['_tmp'] = np.maximum(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0)  # spread
        char = self.get_idym_group()['_tmp'].mean().to_numpy()

        return char

    def _roll_126d(self):
        """Create chars that use 6-month data."""

        if self.created('ami_126d'):
            return

        icolumns = ['mon', 'ret', 'dvol', 'vol', 'turnover_d']
        ocolumns = ['ami_126d', 'zero_trades_126d', 'dolvol_126d', 'dolvol_var_126d', 'turnover_126d',
                    'turnover_var_126d']

        self.data[ocolumns] = self._apply_to_ids(icolumns, 'roll_126d', 6)

    def c_ami_126d(self):
        """Illiquidity. Amihud (2002)"""

        self._roll_126d()
        return self.data['ami_126d']

    def c_zero_trades_126d(self):
        """Zero-trading days (6 months). Liu (2006)"""

        self._roll_126d()
        return self.data['zero_trades_126d']

    def c_turnover_126d(self):
        """Share turnover (JKP). Datar, Naik, and Radcliffe (1998)"""

        self._roll_126d()
        return self.data['turnover_126d']

    def c_turnover_var_126d(self):
        """Volatility of share turnover (JKP). Chordia, Subrahmanyam, and Anshuman (2001)"""

        self._roll_126d()
        return self.data['turnover_var_126d']

    def c_std_turn(self):
        """Volatility of share turnover (GHZ). Chordia, Subrahmanyam, and Anshuman (2001)"""

        cd = self.cd.data
        cd['_tmp'] = cd.vol / cd.shrout
        return self.get_idym_group()._tmp.std().to_numpy()

    def c_dolvol_126d(self):
        """Dollar trading volume (JKP). Brennan, Chordia, and Subrahmanyam (1998)"""

        self._roll_126d()
        return self.data['dolvol_126d']

    def c_dolvol_var_126d(self):
        """Volatility of dollar trading volume (JKP). Chordia, Subrahmanyam, and Anshuman (2001)"""

        self._roll_126d()
        return self.data['dolvol_var_126d']

    def c_std_dolvol(self):
        """Volatility of dollar trading volume (GHZ). Chordia, Subrahmanyam, and Anshuman (2001)"""

        cd = self.cd.data
        cd['_tmp'] = np.log(cd.dvol)
        return self.get_idym_group()._tmp.std().to_numpy()

    def _roll_252d(self):
        """Create chars that use 12-month data."""

        if self.created('rvol_252d'):
            return

        icolumns = ['mon', 'exret', 'mktrf', 'prc_adj', 'vol', 'turnover_d']
        ocolumns = ['rvol_252d', 'mktvol_252d', 'ivol_capm_252d', 'prc_highprc_252d', 'zero_trades_252d',
                    'betadown_252d']

        self.data[ocolumns] = self._apply_to_ids(icolumns, 'roll_252d', 6)

    def c_rmax5_rvol_21d(self):
        """Highest 5 days of return to volatility. Assness et al. (2020)"""

        self.prepare(['rmax5_21d'])
        self._roll_252d()  # to create rvol_252d
        return self.data.rmax5_21d / self.data.rvol_252d

    def c_ivol_capm_252d(self):
        """Idiosyncratic volatility (Org, JKP). Ali, Hwang, and Trombley (2003)"""

        self._roll_252d()
        return self.data['ivol_capm_252d']

    def c_betadown_252d(self):
        """Downside beta. Ang, Chen, and Xing (2006)"""

        self._roll_252d()
        return self.data['betadown_252d']

    def c_prc_highprc_252d(self):
        """52-week high. George and Hwang (2004)"""

        self._roll_252d()
        return self.data['prc_highprc_252d']

    def c_zero_trades_252d(self):
        """Zero-trading days (12 months). Liu (2006)"""

        self._roll_252d()
        return self.data['zero_trades_252d']

    def c_corr_1260d(self):
        """Market correlation. Assness et al. (2020)"""
        cd = self.cd.data
        data = cd.loc[:, ['mon']].copy()
        data['exret_3l'] = self.cd.rolling(cd.exret, 3, 'mean')
        data['mktrf_3l'] = self.cd.rolling(cd.mktrf, 3, 'mean')

        return self._apply_to_ids(data, 'corr_1260d', 1)

    def c_betabab_1260d(self):
        """Frazzini-Pedersen beta. Frazzini and Pedersen (2014)"""

        self.prepare(['corr_1260d'])

        self._roll_252d()  # for rvol_252d and mktvol_252d
        return self.data.corr_1260d * self.data.rvol_252d / self.data.mktvol_252d

    def _reg_weekly_vars(self):
        """Create chars that use weekly returns.
          beta (GHZ), betasq (GHZ), idiovol (GHZ), pricedelay, pricedelay_slope
        """

        if self.created('pricedelay'):
            return

        vars = self.cd[['ret', 'ym']].reset_index()
        vars['yw'] = vars.date.dt.isocalendar().year * 100 + vars.date.dt.isocalendar().week

        vars['wkret'] = np.log(vars.ret + 1)
        gb = vars.groupby(['permno', 'yw'])
        reg = pd.DataFrame(np.exp(gb.wkret.sum()) - 1)
        reg['ym'] = gb.ym.nth(-1).to_numpy()
        del gb
        del vars

        reg['ewret'] = reg.groupby(['yw']).wkret.transform('mean')
        reg['ewret_l1'] = reg['ewret'].shift(1)
        reg['ewret_l2'] = reg['ewret'].shift(2)
        reg['ewret_l3'] = reg['ewret'].shift(3)
        reg['ewret_l4'] = reg['ewret'].shift(4)

        gb = reg.groupby('permno')
        reg.loc[gb.cumcount() < 4, ['ewret_l1', 'ewret_l2', 'ewret_l3', 'ewret_l4']] = np.nan

        gsize = gb.size().to_numpy()
        data = reg[['wkret', 'ewret', 'ewret_l1', 'ewret_l2', 'ewret_l3', 'ewret_l4']].to_numpy()
        res = apply_to_groups_jit(data[:, :2], gsize, rolling_regression, 4, 156, 52, True)
        reg['beta'] = res[:, 1]
        reg['betasq'] = reg['beta'] ** 2
        reg['idiovol'] = res[:, -1]
        r2 = res[:, -2]

        res = apply_to_groups_jit(data, gsize, rolling_regression, 8, 156, 52, True)
        del data
        beta2 = res[:, :-2]
        r2_multi = res[:, -2]

        reg['pricedelay'] = 1 - r2 / r2_multi
        reg['pricedelay_slope'] = (beta2[:, 2:] * np.array([1, 2, 3, 4])).sum(1) / beta2[:, 1:].sum(1)

        reg_ym = reg.reset_index().drop_duplicates(subset=['ym', 'permno'], keep='last')
        reg_ym.set_index(['ym', 'permno'], inplace=True)
        reg_ym.index.names = ('date', 'permno')

        ocolumns = ['beta', 'betasq', 'idiovol', 'pricedelay', 'pricedelay_slope']
        self.data[ocolumns] = reg_ym.loc[:, ocolumns]

    def c_beta(self):
        """Market beta  (GHZ). Fama and MacBeth (1973)"""

        self._reg_weekly_vars()
        return self.data['beta']

    def c_betasq(self):
        """Beta squared (GHZ). Fama and MacBeth (1973)"""

        self._reg_weekly_vars()
        return self.data['betasq']

    def c_idiovol(self):
        """Idiosyncratic volatility (GHZ). Ali, Hwang, and Trombley (2003)"""

        # GHZ use weekly returns.
        self._reg_weekly_vars()
        return self.data['idiovol']

    def c_pricedelay(self):
        """Price delay based on R-squared. Hou and Moskowitz (2005)"""

        self._reg_weekly_vars()
        return self.data['pricedelay']

    def c_pricedelay_slope(self):
        """Price delay based on slopes. Hou and Moskowitz (2005)"""

        self._reg_weekly_vars()
        return self.data['pricedelay_slope']

    def c_baspread(self):
        """Bid-ask spread. Amihud and Mendelson (1986)"""

        cd = self.cd.data
        cd['_tmp'] = (cd.askhi - cd.bidlo) / (0.5 * (cd.askhi + cd.bidlo))
        return self.get_idym_group()._tmp.mean().to_numpy()

    # def c_trend_factor(self):
    #     """Trend factor. Han, Zhou, and Zhu (2016)"""
    #
    #     prc_adj = self.cd['prc_adj'].to_numpy()
    #
    #     avg_prc = np.full([self.cd.data.shape[0], 12], np.nan)
    #     avg_prc[:, 0] = prc_adj  # prc at t
    #     avg_prc[:, 1] = self.cd.rolling(prc_adj, 3, 'mean')  # 3-day mean
    #     avg_prc[:, 2] = self.cd.rolling(prc_adj, 5, 'mean')  # 5-day mean
    #     avg_prc[:, 3] = self.cd.rolling(prc_adj, 10, 'mean')  # 10-day mean
    #     avg_prc[:, 4] = self.cd.rolling(prc_adj, 20, 'mean')  # 20-day mean
    #     avg_prc[:, 5] = self.cd.rolling(prc_adj, 50, 'mean')  # 50-day mean
    #     avg_prc[:, 6] = self.cd.rolling(prc_adj, 100, 'mean')  # 100-day mean
    #     avg_prc[:, 7] = self.cd.rolling(prc_adj, 200, 'mean')  # 200-day mean
    #     avg_prc[:, 8] = self.cd.rolling(prc_adj, 400, 'mean')  # 400-day mean
    #     avg_prc[:, 9] = self.cd.rolling(prc_adj, 600, 'mean')  # 600-day mean
    #     avg_prc[:, 10] = self.cd.rolling(prc_adj, 800, 'mean')  # 800-day mean
    #     avg_prc[:, 11] = self.cd.rolling(prc_adj, 1000, 'mean')  # 1000-day mean
    #
    #     avg_prc /= avg_prc[:, [0]]  # normalized average prices including constant.
    #
    #     ret = self.cd[['ym', 'ret']].reset_index(level=0, drop=True).set_index('ym', append=True)
    #     avg_prc = pd.DataFrame(avg_prc, index=ret.index)
    #
    #     ret = np.log(ret + 1)
    #     ret = np.exp(ret.groupby(level=[0, 1]).sum()) - 1
    #     avg_prc = avg_prc.groupby(level=[0, 1]).nth(-1)  # last() for old behavior
    #     sample = pd.concat([ret, avg_prc.groupby(level=0).shift(1)], axis=1)
    #
    #     beta = {}
    #     for ym, sample_ in sample.groupby('ym'):
    #         sample_.dropna(inplace=True)
    #         y = sample_.iloc[:, 0]
    #         X = sample_.iloc[:, 1:]
    #         if X.shape[0] < X.shape[1]:  # singular
    #             beta[ym] = np.full(X.shape[1], np.nan)
    #         else:
    #             beta[ym] = (np.linalg.solve(X.T @ X, X.T @ y)).reshape(X.shape[1])
    #
    #     del sample
    #
    #     beta = pd.DataFrame.from_dict(beta, orient='index')
    #     est_beta = beta.rolling(12).mean()  # 12-month mean of beta estimates.
    #
    #     char = []
    #     for ym, avg_prc_ in avg_prc.groupby('ym'):
    #         char.append(avg_prc_.dot(est_beta.loc[ym]))  # Return prediction
    #
    #     return pd.concat(char).sort_index().to_numpy()  # ym/permno sorted on permno/ym

    def c_trend_factor(self):
        """Trend factor. Han, Zhou, and Zhu (2016)"""

        ret = self.cd[['ym', 'ret']].reset_index(level=0, drop=True).set_index('ym', append=True)
        ret = np.log(ret + 1)
        ret['idx'] = np.arange(ret.shape[0])
        gb = ret.groupby(level=[0, 1])
        ret = np.exp(gb['ret'].sum()) - 1
        idx = gb['idx'].last().to_numpy()
        del gb

        prc_adj = self.cd['prc_adj'].to_numpy()

        avg_prc = np.full([ret.shape[0], 12], np.nan)
        avg_prc[:, 0] = prc_adj[idx]  # prc at t
        avg_prc[:, 1] = self.cd.rolling(prc_adj, 3, 'mean')[idx]  # 3-day mean
        avg_prc[:, 2] = self.cd.rolling(prc_adj, 5, 'mean')[idx]  # 5-day mean
        avg_prc[:, 3] = self.cd.rolling(prc_adj, 10, 'mean')[idx]  # 10-day mean
        avg_prc[:, 4] = self.cd.rolling(prc_adj, 20, 'mean')[idx]  # 20-day mean
        avg_prc[:, 5] = self.cd.rolling(prc_adj, 50, 'mean')[idx]  # 50-day mean
        avg_prc[:, 6] = self.cd.rolling(prc_adj, 100, 'mean')[idx]  # 100-day mean
        avg_prc[:, 7] = self.cd.rolling(prc_adj, 200, 'mean')[idx]  # 200-day mean
        avg_prc[:, 8] = self.cd.rolling(prc_adj, 400, 'mean')[idx]  # 400-day mean
        avg_prc[:, 9] = self.cd.rolling(prc_adj, 600, 'mean')[idx]  # 600-day mean
        avg_prc[:, 10] = self.cd.rolling(prc_adj, 800, 'mean')[idx]  # 800-day mean
        avg_prc[:, 11] = self.cd.rolling(prc_adj, 1000, 'mean')[idx]  # 1000-day mean

        avg_prc /= avg_prc[:, [0]]  # normalized average prices including constant.

        avg_prc = pd.DataFrame(avg_prc, index=ret.index)
        sample = pd.concat([ret, avg_prc.groupby(level=0).shift(1)], axis=1)

        beta = {}
        for ym, sample_ in sample.groupby('ym'):
            sample_.dropna(inplace=True)
            y = sample_.iloc[:, 0]
            X = sample_.iloc[:, 1:]
            if X.shape[0] < X.shape[1]:  # singular
                beta[ym] = np.full(X.shape[1], np.nan)
            else:
                beta[ym] = (np.linalg.solve(X.T @ X, X.T @ y)).reshape(X.shape[1])

        del sample

        beta = pd.DataFrame.from_dict(beta, orient='index')
        est_beta = beta.rolling(12).mean()  # 12-month mean of beta estimates.

        char = []
        for ym, avg_prc_ in avg_prc.groupby('ym'):
            char.append(avg_prc_.dot(est_beta.loc[ym]))  # Return prediction

        return pd.concat(char).sort_index().to_numpy()  # ym/permno sorted on permno/ym


################################################################################
#
# MERGE
#
################################################################################

class Merge(FCPanel):
    """Class to generate firm characteristics from a combined dataset of crspm, crspd, funda, and fundq.

    The firm characteristics defined in this class can be viewed using :meth:`.show_available_chars`.

    **Methods**

    .. autosummary::
        :nosignatures:

        preprocess
        postprocess

    Firm characteristic generation methods have a name like ``c_characteristic()``.
    """

    def __init__(self, alias=None):
        super().__init__(alias)

    def preprocess(self, crspm=None, crspd=None, funda=None, fundq=None, delete_data=True):
        """Merge crspm, crspd, funda, and fundq.

        The crspd, funda, and fundq are left-joined to crspm, and the resulting data has index = date/permno.
        The frequency of the final data is the same as the frequency of crspm data. This method also checks
        if "ingredient" firm characteristics have been generated and generates them if necessary.

        Args:
            crspm: :class:`CRSPM` instance.
            crspd: :class:`CRSPD` instance.
            funda: :class:`FUNDA` instance.
            fundq: :class:`FUNDQ` instance.
            delete_data: True to delete the data of crspm, crspd, funda, and fundq after merge to save memory.
        """

        elapsed_time('Preprocessing Merge...')

        ###############################
        # Prepare characteristics required to generate characteristics here..
        ###############################
        if 'age' in self.char_map:
            funda.prepare(['age'])

        if 'mispricing_perf' in self.char_map:
            funda.prepare(['o_score', 'gp_at'])
            fundq.prepare(['niq_at'])
            crspm.prepare(['ret_12_1'])

        if 'mispricing_mgmt' in self.char_map:
            funda.prepare(['oaccruals_at', 'noa_at', 'at_gr1', 'ppeinv_gr1a'])
            crspm.prepare(['chcsho_12m', 'eqnpo_12m'])

        if 'qmj' in self.char_map:
            funda.prepare(['gp_at', 'ni_be', 'ocf_at', 'oaccruals_at', 'o_score', 'z_score'])
            funda['ni_at'] = funda['ib'] / funda['at_']
            funda['gp_sale'] = funda['gp'] / funda['sale']
            funda['debt_at'] = funda['debt'] / funda['at_']
            funda['gpoa_ch5'] = funda.diff(funda['gp_at'], 5)
            funda['roe_ch5'] = funda.diff(funda['ni_be'], 5)
            funda['roa_ch5'] = funda.diff(funda['ni_at'], 5)
            funda['cfoa_ch5'] = funda.diff(funda['ocf_at'], 5)
            funda['gmar_ch5'] = funda.diff(funda['gp_sale'], 5)
            funda['roe_std'] = funda.rolling(funda['ni_be'], 5, 'std')

            fundq['roeq_std'] = fundq.rolling(fundq['ibq'] / fundq['be'], 20, 'std', min_n=12)

            crspd.prepare(['betabab_1260d'])

        ###############################
        # Merge data
        ###############################
        char_map = self.char_map
        # At least one of crspm or crspd should be provided.
        if (crspm is not None) and (crspd is not None):  # Both exist.
            self.copy_from(crspm)
            char_map.update(crspm.char_map)
            if delete_data:
                del crspm.data

            self.merge(crspd.data, how='left')
            char_map.update(crspd.char_map)
            if delete_data:
                del crspd.data
        elif crspm is not None and crspd is None:  # Only crspm.
            self.copy_from(crspm)
            char_map.update(crspm.char_map)
            if delete_data:
                del crspm.data
        elif crspm is None and crspd is not None:  # Only crspd.
            self.copy_from(crspd)
            char_map.update(crspd.char_map)
            if delete_data:
                del crspd.data
        else:
            raise ValueError('At least one of CRSPM or CRSPD should be provided.')
        self.char_map = char_map

        if funda is not None:
            self.merge(funda.data, on=['date', 'gvkey'], how='left')
            self.data.rename(columns={'age': 'comp_age'}, inplace=True)
            char_map.update(funda.char_map)
            if delete_data:
                del funda.data

        if fundq is not None:
            self.merge(fundq.data, on=['date', 'gvkey'], how='left')
            char_map.update(fundq.char_map)
            if delete_data:
                del fundq.data

        # Add future return
        self.data['futret'] = self.futret(self.data.exret)
        elapsed_time('Merge preprocessed.')

    def postprocess(self):
        """Postprocess data.

        This method deletes temporary variable columns (columns starting with '\\_') and replaces infinite
        values with nan.
        """

        elapsed_time('Postprocessing Merge...')
        md = self.data

        # drop temporary columns that start with '_'
        del_cols = [col for col in md if col[0] == '_']
        drop_columns(md, del_cols)

        char_cols = list(self.get_char_list())
        self.inspect_data(char_cols, option=['summary', 'nans'])

        log('Replacing inf with nan...')
        for col in char_cols:
            isinf = (md[col] == np.inf) | (md[col] == -np.inf)
            if isinf.any():
                md.loc[isinf, col] = np.nan

        self.inspect_data(char_cols, option=['stats'])
        elapsed_time('Merge postprocessed.')

    #######################################
    # Characteristics
    #######################################
    def c_age(self):
        """Firm age. Jiang, Lee, and Zhang (2005)"""

        # Update age as the maximum of age from funda and age from crspm.
        md = self.data
        md['age'] = (self.get_row_count() + 1) * 12 / self.freq  # age from crspm

        if 'comp_age' in md:
            @njit(cache=True)
            def fcn(data):
                gap = np.nanmax(data[:, 1] - data[:, 0])  # max(comp_age - crsp_age)
                if gap > 0:
                    return data[:, 0] + gap  # crsp_age + max(comp_age - crsp_age)
                else:
                    return data[:, 0]  # crsp_age

            md['age'] = self.apply_to_ids(['age', 'comp_age'], fcn, 1)
            # md['age'] = group_and_apply(md.loc[:, ['age', 'comp_age']], 'permno', fcn, values=True)
            del md['comp_age']
            # The code below can yield younger ages than previous ages when comp_age is missing: if funda data exists
            # from 2000.01 to 2020.12 and crsp data from 2001.01 to 2022.12, the age from 2021.01 will be younger.
            # md['age'] = np.nanmax(md.loc[:, ['age', 'comp_age']], axis=1)

        return md['age']

    def c_mispricing_perf(self):
        """Mispricing factor: Performance. Stambaugh and Yuan (2016)"""

        tmp = self.apply_to_dates(['o_score', 'ret_12_1', 'gp_at', 'niq_at'], rank, 4, True, True)

        tmp[:, 0] = 1 - tmp[:, 0]  # tmp.o_score = 1 - tmp.o_score
        return np.nanmean(tmp, axis=1)

    def c_mispricing_mgmt(self):
        """Mispricing factor: Management. Stambaugh and Yuan (2016)"""

        tmp = self.apply_to_dates(['chcsho_12m', 'eqnpo_12m', 'oaccruals_at', 'noa_at', 'at_gr1', 'ppeinv_gr1a'], rank,
                                  6, False, True)

        if config.replicate_jkp:
            tmp[:, 1] = 1 - tmp[:, 1]  # tmp.eqnpo_12m = 1 - tmp.eqnpo_12m

        char = np.where(np.sum(~np.isnan(tmp), axis=1) < 3, np.nan, np.nanmean(tmp, axis=1))
        return char

    @staticmethod
    @njit(cache=True)
    def _zrank(x):
        """Normalized rank.
        """

        rx = rank(x)
        if x.ndim == 1:
            return (rx - np.nanmean(rx)) / nanstd(rx)

        isna = np.isnan(rx)
        ry = np.zeros(rx.shape[0])
        for j in range(x.shape[1]):
            rx_j = (rx[:, j] - np.nanmean(rx[:, j])) / nanstd(rx[:, j])
            ry += np.where(np.isnan(rx_j), 0, rx_j)

        ry /= x.shape[1] - np.sum(isna, axis=1)
        ry = rank(ry)
        return (ry - np.nanmean(ry)) / nanstd(ry)

    def c_qmj_growth(self):
        """Quality minus Junk: Growth. Assness, Frazzini, and Pedersen (2018)"""

        grow = ['gpoa_ch5', 'roe_ch5', 'roa_ch5', 'cfoa_ch5', 'gmar_ch5']
        return self.apply_to_dates(grow, self._zrank, 1)

    def c_qmj_prof(self):
        """Quality minus Junk: Profitability. Assness, Frazzini, and Pedersen (2018)"""

        prof = self[['gp_at', 'ni_be', 'ni_at', 'ocf_at', 'gp_sale', 'oaccruals_at']]
        prof['oaccruals_at'] *= -1
        return self.apply_to_dates(prof, self._zrank, 1)

    def c_qmj_safety(self):
        """Quality minus Junk: Safety. Assness, Frazzini, and Pedersen (2018)"""

        safe = -self[['betabab_1260d', 'debt_at', 'o_score']]
        safe['z_score'] = self['z_score']
        safe['evol'] = -self['roeq_std'].fillna(self['roe_std'] / 2)
        return self.apply_to_dates(safe, self._zrank, 1)

    def c_qmj(self):
        """Quality minus Junk: Composite. Assness, Frazzini, and Pedersen (2018)"""

        self.prepare(['qmj_growth', 'qmj_prof', 'qmj_safety'])

        qmj = (self['qmj_prof'] + self['qmj_growth'] + self['qmj_safety']) / 3
        return self.apply_to_dates(qmj, self._zrank, 1)


if __name__ == '__main__':
    os.chdir('../')
