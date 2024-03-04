"""This module defines WRDS class that is used to download and handle WRDS data.

    .. autosummary::
        :nosignatures:

        WRDS
"""

import shutil
import pandas as pd
import datetime as dt
import sqlite3
import wrds
import asyncio

from pyanomaly.globals import *
from pyanomaly.datatools import merge, to_month_end
from pyanomaly.datatools import inspect_data

################################################################################
#
# Class to download/handle WRDS data.
#
# Useful links
#  - CRSP overview
#     https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/
#  - CRSP-COMPUSTST merge overview
#     https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/crspcompustat-merged-ccm/wrds-overview-crspcompustat-merged-ccm/
#  - CRSP annual update table list
#     https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_indexes/
#
# Exchcd
# -2: Halted by the NYSE or AMEX
# -1: Suspended by the NYSE, AMEX, or NASDAQ
#  0: Not Trading on NYSE, AMEX, or NASDAQ
#  1: New York Stock Exchange
#  2: American Stock Exchange
#  3: NASDAQ
#
# Shrcd (https://wrds-www.wharton.upenn.edu/data-dictionary/form_metadata/crsp_a_stock_msf_identifyinginformation/shrcd/)
#  1st digit
#  1: Ordinary Common Shares
#  2: Certificates
#  3: ADRs (American Depository Receipts)
#  4: SBIs (Shares of Beneficial Interest)
#  7: Units (Depository Units, Units of Beneficial Interest, Units of Limited Partnership Interest, Depository Receipts, etc.)
#  2nd digit
#  0: Securities which have not been further defined.
#  1: Securities which need not be further defined.
#  2: Companies incorporated outside the US
#  3: Americus Trust Components (Primes and Scores).
#  4: Closed-end funds.
#  5: Closed-end fund companies incorporated outside the US
#  8: REIT's (Real Estate Investment Trusts).
################################################################################


class WRDS:
    """Class to download/handle WRDS data.

    Args:
        wrds_username: WRDS username. Required only when downloading data: can be set to None when reading
            data from files.

    **Attributes**

    Attributes:
        db: A ``wrds`` object to connect to WRDS database.

    **Methods for Data Download**

    .. autosummary::
        :nosignatures:

        download_table
        download_table_async
        download_sf
        download_seall
        download_funda
        download_fundq
        download_secd
        download_g_secd
        download_all

    **Other Methods**

    .. autosummary::
        :nosignatures:

        create_pgpass_file
        merge_sf_with_seall
        add_gvkey_to_crsp
        create_crsp_comp_linktable
        add_gvkey_to_crsp_cusip
        preprocess_crsp
        get_risk_free_rate
        convert_fund_currency_to_usd
        save_data
        read_data
        save_as_csv

    References:
        CRSP overview: https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/

        CRSP-Compustat merge: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/crspcompustat-merged-ccm/wrds-overview-crspcompustat-merged-ccm/

        CRSP annual update tables: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_indexes/

        shrcd: https://wrds-www.wharton.upenn.edu/data-dictionary/form_metadata/crsp_a_stock_msf_identifyinginformation/shrcd/

        exchcd: https://wrds-www.wharton.upenn.edu/data-dictionary/form_metadata/crsp_a_stock_msf_identifyinginformation/exchcd/
    """
    def __init__(self, wrds_username=None):
        if wrds_username:
            self.db = wrds.Connection(wrds_username=wrds_username)

    def create_pgpass_file(self):
        """Create pgpass file.

        Need to be called only once (after logging in to WRDS for the first time using passwords).
        Once pgpass file is created, password is not required when connecting to WRDS.
        """

        self.db.create_pgpass_file()

    ################################################################################
    #
    # DATA DOWNLOAD
    #
    ################################################################################
    def _convert_types(self, data, src_tables):
        """Convert datatypes of `data` based on the table descriptions from WRDS.

        * Permno -> Int64 (Pandas nullable int).
        * Double precision -> `config.float_type`.
        * Date -> Pandas datetime.
        * Varchar -> Pandas string (consumes significantly less memory than object type).

        Args:
            data: DataFrame.
            src_tables: list of (library, table) that are used to create `data`.

        Returns:
            DataFrame. type-converted `data`.
        """

        desc = [self.db.describe_table(*src) for src in src_tables]  # get table descriptions
        desc = pd.concat(desc).drop_duplicates(subset=['name'])

        print(desc)
        for col in data:
            data_type = str(desc.loc[desc.name == col, 'type'].values[0])
            if col in ('permno', 'lpermno'):
                data[col] = data[col].astype('Int64')
            elif (((data_type == 'DOUBLE_PRECISION') or (data_type[:7] == 'NUMERIC')) and
                  data[col].dtype != config.float_type):
                data[col] = data[col].astype(config.float_type, errors='ignore')
            elif (data_type == 'DATE') and not pd.api.types.is_datetime64_ns_dtype(data[col]):
                data[col] = pd.to_datetime(data[col])
            elif (data_type[:7] == 'VARCHAR') and data[col].dtype == 'object':
                data[col] = data[col].astype('string')

        print(data.info(True))
        return data

    def download_table(self, library, table, obs=-1, offset=0, columns=None, coerce_float=None, date_cols=None,
                       index_col=None, sort_col=None):
        """Download a table from WRDS library.

        This is a wrapping function of ``wrds.get_table()``. The queried table is saved to
        ``config.input_dir/library/table``.

        Args:
            library: WRDS library. e.g., crsp, comp, ...
            table: A table in `library`.
            obs: See ``wrds.get_table()``.
            offset: See ``wrds.get_table()``.
            columns: See ``wrds.get_table()``.
            coerce_float: See ``wrds.get_table()``.
            date_cols: See ``wrds.get_table()``.
            index_col: (List of) column(s) to be set as index.
            sort_col: (List of) column(s) to sort data on.
        """

        elapsed_time(f'Downloading {library}.{table}...')
        data = self.db.get_table(
            library=library,
            table=table,
            obs=obs,
            offset=offset,
            columns=columns,
            date_cols=date_cols)
        data = self._convert_types(data, [(library, table)])

        self.save_data(data, table, library, index_col, sort_col)
        elapsed_time(f'Download complete. Size: {data.shape}')

    def _generate_sql(self, library, table, sql, date_col):
        if not sql or ('from' not in sql.lower()):
            select = sql or '*'

            sql = f"""
                SELECT {select}
                FROM 
                    {library}.{table}
                WHERE {date_col} between '{{}}' and '{{}}'
                """
        print(sql)
        return sql

    def _download_table_async(self, table, sql, sdate, edate):
        log(f'Downloading... {table}: {sdate}-{edate}')
        # await asyncio.sleep(.01)

        data = self.db.raw_sql(sql.format(sdate, edate))
        if not data.empty:
            self.save_data(data, f"{table}_{sdate.replace('-', '')}_{edate.replace('-', '')}", 'tmp')

        log(f"Download complete. {table}: {sdate}-{edate}. Num. records: {len(data)}")

    def download_table_async(self, library, table, sql=None, date_col=None, sdate=None, edate=None, interval=5,
                             src_tables=None, run_in_executer=True, index_col=None, sort_col=None):
        """Asynchronous download of a WRDS table.

        This method splits the total period into `interval` years and downloads data of each sub-period
        asynchronously. If download fails, it can be started from the failed date: already downloaded files will be
        gathered together. This method allow us to download a large table, e.g., crsp.dsf, reliably without connection
        timeout and consumes much less memory than :meth:`download_table`.
        The queried table is saved to ``config.input_dir/library/table``.

        Args:
            library: WRDS library.
            table: WRDS table. If a complete query is given in `sql`, this can be any name: used only as the file name
                when saving the data.
            sql: String of the fields to select or a complete query statement. See below.
            date_col: Date field on which downloaing will be split. Ignored if `sql` is a complete query statement.
            sdate: Start date ('yyyy-mm-dd'). If None, '1900-01-01'.
            edate: End date. If None, today.
            interval: Sub-period size in years.
            src_tables: List of (library, table) that are used in the query. The `src_tables` are used to get data types
                of the fields. When data is selected from a single table, `library.table`, this can be set to None.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
            index_col: (List of) column(s) to be set as index.
            sort_col: (List of) column(s) to sort data on.

        NOTE:
            For a small table, this can be slower than :py:meth:`download_table`.
            Table should have a date field (`date_col`) to split the period.

        Examples:

            Instantiate WRDS.

            >>> wrds = WRDS('user_name')

            Download crsp.msf.

            >>> wrds.download_table_async('crsp', 'msf', date_col='date')

            Download 'permno', 'prc', and 'exchcd' fields from crsp.msf.

            >>> sql = 'permno, prc, exchcd'
            >>> wrds.download_table_async('crsp', 'msf', sql, 'date')

            Download crsp.msf merged with crsp.msenames. When `sql` is a complete query statement as below,
            it should contain 'WHERE [`date_col`] BETWEEN {} and {}' for asynchronous download.

            >>> sql = '''
            ...        SELECT a.*, b.shrcd, b.exchcd, b.siccd
            ...        FROM crsp.msf as a
            ...        LEFT JOIN crsp.msenames as b
            ...        ON a.permno = b.permno
            ...        AND b.namedt <= a.date
            ...        AND a.date <= b.nameendt
            ...        WHERE a.date BETWEEN '{{}}' and '{{}}'
            ...        ORDER BY a.permno, a.date
            ...     '''
            >>> src_tables = [('crsp', 'msf'), ('crsp', 'msenames')]
            >>> wrds.download_table_async('crsp', 'msf', sql, src_tables=src_tables)
        """

        elapsed_time(f'Asynchronous download: {table}')

        # Make a list of sub-periods.
        ldate = edate or str(dt.datetime.now())[:10]  # last date
        sdate = sdate or '1900-01-01'  # sub-period start date
        edate = min(str(pd.Timestamp(sdate) + pd.DateOffset(years=interval, days=-1))[:10], ldate)  # sub-period end date

        dates = []
        while sdate < ldate:
            dates.append((sdate, edate))
            sdate = str(int(sdate[:4]) + interval) + sdate[4:]
            edate = min(str(int(edate[:4]) + interval) + edate[4:], ldate)

        # Generate query statement.
        sql = self._generate_sql(library, table, sql, date_col)

        # Download and save data for each sub-period.
        if (date_col is None) and (sql and '{}' not in sql):  # Download at once.
            data = self.db.raw_sql(sql)
        else:
            if run_in_executer:  # Asynchronous download.
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                aws = []
                for i, (sdate, edate) in enumerate(dates):
                    aws.append(loop.run_in_executor(None, self._download_table_async, table, sql, sdate, edate))
                    if (i and (i % 5 == 0)) or (edate == ldate):  # 5 workers at a time.
                        loop.run_until_complete(asyncio.gather(*aws))
                        aws = []
            else:
                for sdate, edate in dates:
                    self._download_table_async(table, sql, sdate, edate)

            # Read saved data files and save them as one big file.
            log('Gathering files...')
            read_file = pd.read_parquet if config.file_format == 'parquet' else pd.read_pickle
            tmp_dir = config.input_dir + 'tmp'  # sub-period files are temporarily saved here.
            data = []
            for f in os.listdir(tmp_dir):
                fpath = os.path.join(tmp_dir, f)
                if os.path.isfile(fpath) and (f[:len(table)] == table):
                    data.append(read_file(fpath))
            if len(data) != 0:
                data = pd.concat(data)

            for file in os.scandir(tmp_dir):
                os.remove(file.path)

        # type conversion
        if src_tables is None:
            src_tables = [(library, table)]
        data = self._convert_types(data, src_tables)

        self.save_data(data, table, library, index_col, sort_col)
        elapsed_time(f'Download complete: {table}')

    def download_sf(self, sdate=None, edate=None, monthly=True, run_in_executer=True):
        """ Download crsp.m(d)sf joined with crsp.m(d)senames.

        Downloaded data has index = date/permno and is sorted on permno/date.

        Args:
            sdate: Start date, e.g., '2000-01-01'. Set to None to download all data.
            edate: End date. Set to None to download all data.
            monthly: If True download msf else dsf.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """

        if monthly:
            fields = 'a.*, b.shrcd, b.exchcd, b.siccd, b.ncusip'
            sf = 'msf'
            senames = 'msenames'
        else:  # for dsf, download only necessary columns to reduce file size.
            fields = 'a.permno, a.permco, a.date, bidlo, askhi, prc, ret, vol, shrout, cfacpr, cfacshr, b.shrcd, b.exchcd'
            sf = 'dsf'
            senames = 'dsenames'

        sql = f"""
            SELECT {fields}
            FROM crsp.{sf} as a
            LEFT JOIN crsp.{senames} as b
            ON a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date <= b.nameendt
            WHERE a.date between '{{}}' and '{{}}'
            ORDER BY a.permno, a.date
        """

        src_tables = [('crsp', sf), ('crsp', senames)]
        self.download_table_async('crsp', sf, sql, sdate=sdate, edate=edate, src_tables=src_tables,
                                  run_in_executer=run_in_executer,
                                  index_col=['date', 'permno'], sort_col=['permno', 'date'])

    def download_seall(self, sdate=None, edate=None, monthly=True, run_in_executer=True):
        """Download delist and dividend info from crsp.m(d)seall.

        Delist can be obtained from either mseall or msedelist. We use mseall since it contains exchcd, which is used
        when replacing missing dlret.
        The shrcd and exchcd in mseall are usually those before halting/suspension. If a stock in NYSE is halted, exchcd
        in msenames can be -2, whereas that in mseall is 1.
        The downloaded fields are: permno, date, dlret, dlstcd, shrcd, exchcd, distcd, divamt.
        Downloaded data has index = date/permno and is sorted on permno/date.


        Args:
            sdate: Start date, e.g., '2000-01-01'. Set to None to download all data.
            edate: End date. Set to None to download all data.
            monthly: If True download mseall else dseall.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """

        seall = 'mseall' if monthly else 'dseall'

        sql = f"""
            SELECT distinct permno, date, 
            dlret, dlstcd, shrcd, exchcd, 
            distcd, divamt  
            FROM crsp.{seall}
            WHERE date between '{{}}' and '{{}}'
        """

        self.download_table_async('crsp', seall, sql, sdate=sdate, edate=edate, run_in_executer=run_in_executer,
                                  index_col=['date', 'permno'], sort_col=['permno', 'date'])

    def download_funda(self, sdate=None, edate=None, run_in_executer=True):
        """Download comp.funda.

        Downloaded data has index = datadate/gvkey and is sorted on gvkey/datadate.

        Args:
            sdate: Start date, e.g., '2000-01-01'. Set to None to download all data.
            edate: End date. Set to None to download all data.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """

        sql = f"""
            SELECT 
                cusip, c.gvkey, datadate, c.cik, sic, naics, fyear,
                sale, revt, cogs, xsga, dp, xrd, xad, ib, gp, ebitda, ebit, nopi, spi, pi, txp, ni, txfed, txfo, txt, xint,
                capx, oancf, dvt, ob, gdwlia, gdwlip, gwo,
                rect, act, che, ppegt, invt, at, aco, intan, ao, ppent, gdwl, fatb, fatl,
                lct, dlc, dltt, lt, dm, dcvt, cshrc, dcpstk, pstk, ap, lco, lo, drc, drlt, txdi,
                ceq, scstkc, emp, csho, prcc_f,
                oibdp, oiadp, mii, xopr, xlr, xi, f.do, xido, ibc, dpc, xidoc,
                fincf, fiao, txbcof, ds, dltr, dlcch, prstkc, sstk, dv, dvc,
                ivao, ivst, re, txditc, txdb, itcb, seq, pstkrv, pstkl, mib, icapt, dltis, ajex, ppenb, ppenls, 
                curcd
                /* wcapt, ltdch, purtshr */
            FROM 
                comp.company as c,
                comp.funda as f
            WHERE c.gvkey = f.gvkey
            AND f.indfmt = 'INDL' AND f.datafmt = 'STD' AND f.popsrc = 'D' AND f.consol = 'C'
            AND datadate between '{{}}' and '{{}}'
            ORDER BY c.gvkey, datadate
            """

        src_tables = [('comp', 'funda'), ('comp', 'company')]
        self.download_table_async('comp', 'funda', sql, sdate=sdate, edate=edate, src_tables=src_tables,
                                  run_in_executer=run_in_executer,
                                  index_col=['datadate', 'gvkey'], sort_col=['gvkey', 'datadate'])

    def download_fundq(self, sdate=None, edate=None, run_in_executer=True):
        """Download comp.fundq.

        Downloaded data has index = datadate/gvkey and is sorted on gvkey/datadate.

        Args:
            sdate: Start date, e.g., '2000-01-01'. Set to None to download all data.
            edate: End date. Set to None to download all data.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """

        sql = f"""
            SELECT 
                cusip, c.gvkey, datadate, c.cik, sic, naics, fyr, fyearq, fqtr,
                saleq, revtq, cogsq, xsgaq, dpq, xrdq, ibq, nopiq, spiq, piq, txpq, niq, txtq, xintq,
                gdwliaq, gdwlipq,
                rectq, actq, cheq, ppegtq, invtq, atq, acoq, intanq, aoq, ppentq, gdwlq,
                lctq, dlcq, dlttq, ltq, pstkq, apq, lcoq, loq, drcq, drltq, txdiq,
                ceqq, cshoq, prccq, rdq, pstkrq, aqaq, aqdq, aqepsq, aqpq, 
                cibegniq, cicurrq, ciderglq, 
                oibdpq, oiadpq, miiq, xoprq, xiq, doq, xidoq,
                ivaoq, ivstq, req, txditcq, txdbq, seqq, mibq, icaptq, ajexq, cshprq, epspxq, 
                saley, revty, cogsy, xsgay,
                dpy, xrdy, iby, nopiy, spiy, piy, niy, txty, xinty, capxy, oancfy, gdwliay, gdwlipy,
                txdiy, scstkcy, oibdpy, oiadpy, miiy, xopry, xiy, doy, xidoy, ibcy, dpcy, xidocy, 
                fincfy, fiaoy, txbcofy, dltry, dlcchy, prstkcy, sstky, dvy, dltisy
                afudciy, amcy, aolochy, apalchy, aqay, aqcy, aqdy, aqepsy, aqpy,  
                cdvcy, chechy, cibegniy, cicurry, cidergly, wcapq, 
                curcdq
            FROM 
                comp.company as c,
                comp.fundq as f
            WHERE c.gvkey = f.gvkey
            AND f.indfmt = 'INDL' AND f.datafmt = 'STD' AND f.popsrc = 'D' AND f.consol = 'C'
            AND datadate between '{{}}' and '{{}}'
            ORDER BY c.gvkey, datadate
            """

        src_tables = [('comp', 'fundq'), ('comp', 'company')]
        self.download_table_async('comp', 'fundq', sql, sdate=sdate, edate=edate, src_tables=src_tables,
                                  run_in_executer=run_in_executer,
                                  index_col=['datadate', 'gvkey'], sort_col=['gvkey', 'datadate'])

    def download_secd(self, sdate=None, edate=None, run_in_executer=True):
        """Download comp.secd.

        Downloaded data has index = datadate/gvkey and is sorted on gvkey/datadate.

        Args:
            sdate: Start date, e.g., '2000-01-01'. Set to None to download all data.
            edate: End date. Set to None to download all data.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """
        sql = 'gvkey, datadate, prccd, ajexdi, cshoc, iid'
        self.download_table_async('comp', 'secd', sql, 'datadate', sdate, edate,
                                  run_in_executer=run_in_executer,
                                  index_col=['datadate', 'gvkey'], sort_col=['gvkey', 'datadate'])

    def download_g_secd(self, sdate=None, edate=None, run_in_executer=True):
        """Download comp.g_secd.

        Downloaded data has index = datadate/gvkey and is sorted on gvkey/datadate.

        Args:
            sdate: Start date, e.g., '2000-01-01'. Set to None to download all data.
            edate: End date. Set to None to download all data.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """
        sql = 'gvkey, datadate, prccd, ajexdi, cshoc, iid'
        self.download_table_async('comp', 'g_secd', sql, 'datadate', sdate, edate,
                                  run_in_executer=run_in_executer,
                                  index_col=['datadate', 'gvkey'], sort_col=['gvkey', 'datadate'])

    def download_all(self, run_in_executer=True):
        """Download all tables.

        Currently, this method downloads the following tables:

        * comp.funda
        * comp.fundq
        * comp.exrt_dly
        * crsp.msf (merged with crsp.msenames)
        * crsp.dsf (merged with crsp.dsenames)
        * crsp.mseall
        * crsp.dseall
        * crsp.ccmxpf_linktable
        * crsp.mcti
        * ff.factors_monthly
        * ff.factors_daily

        Args:
            run_in_executer: If True, download concurrently. Faster (if network speed is high) but memory hungrier.
        """

        self.download_funda(run_in_executer=run_in_executer)
        self.download_fundq(run_in_executer=run_in_executer)
        self.download_table('comp', 'exrt_dly', date_cols=['datadate'])

        self.download_sf(monthly=True, run_in_executer=run_in_executer)
        self.download_sf(monthly=False, run_in_executer=run_in_executer)
        self.download_seall(monthly=True, run_in_executer=run_in_executer)
        self.download_seall(monthly=False, run_in_executer=run_in_executer)
        self.download_table('crsp', 'mcti', date_cols=['caldt'])
        self.download_table('ff', 'factors_monthly', date_cols=['date', 'dateff'])
        self.download_table('ff', 'factors_daily', date_cols=['date'])

        try:
            self.download_table('crsp', 'ccmxpf_linktable', date_cols=['linkdt', 'linkenddt'])
        except Exception as e:
            err(f'Failed to download ccmxpf_linktable: {e}')
            log('Linktable will be created internally.')
            self.download_table('comp', 'security')
            self.create_crsp_comp_linktable()

    ################################################################################
    #
    # UTILITIES
    #
    ################################################################################
    @staticmethod
    def merge_sf_with_seall(monthly=True, fill_method=1):
        """Merge m(d)sf with m(d)seall.

        This method adjusts m(d)sf return ('ret') using m(d)seall delist return ('dlret'). The adjusted return replaces
        'ret' and 'dlret' column is added to m(d)sf.
        For msf, this method also adds cash dividend column, 'cash_div', to msf.

        Args:
            monthly: If True, merge msf with mseall; else, merge dsf with dseall.
            fill_method: Method to fill missing dlret. 0: don't fill, 1: JKP code, or 2: GHZ code. Default to 1.

                - fill_method = 1:

                    - `dlret` = -0.30 if `dlstcd` is 500 or between 520 and 584.
                - fill_method = 2:

                    - `dlret` = -0.35 if `dlstcd` is 500 or between 520 and 584, and `exchcd` is 1 or 2.
                    - `dlret` = -0.55 if `dlstcd` is 500 or between 520 and 584, and `exchcd` is 3.

        Returns:
            m(d)sf with adjusted return (and cash dividend).

        Note:
            The msenames can be missing when a firm is delisted, resulting in missing shrcd/exchcd in m(d)sf.
            Missing shrcd and exchcd are filled with the latest values.

        References:
            Delist codes: http://www.crsp.com/products/documentation/delisting-codes

        """

        f = 'm' if monthly else 'd'
        elapsed_time(f'Merging {f}sf with {f}seall...')

        sf = WRDS.read_data(f'{f}sf')
        seall = WRDS.read_data(f'{f}seall').reset_index()

        if monthly:
            gb = sf.reset_index().groupby('permno')
            sf['shrcd'] = gb.shrcd.ffill().to_numpy()
            sf['exchcd'] = gb.exchcd.ffill().to_numpy()

        #######################################
        # DLRET
        #######################################
        delist = seall[['permno', 'date', 'dlret', 'dlstcd', 'shrcd', 'exchcd']].dropna(subset=['dlstcd']).drop_duplicates()
        log(f'{f}sf shape: {sf.shape}, delist{f} shape: {delist.shape}')

        # Fill missing delist return
        log(f'Missing dlret: {delist.dlret.isna().sum()}')
        if fill_method == 1:  # from JKP SAS code
            ridx = delist.dlret.isnull() & ((delist.dlstcd == 500) | ((delist.dlstcd >= 520) & (delist.dlstcd <= 584)))
            delist['dlret'] = delist['dlret'].mask(ridx, -0.30)
        elif fill_method == 2:  # from GHZ SAS code
            ridx = delist.dlret.isnull() & ((delist.dlstcd == 500) | ((delist.dlstcd >= 520) & (delist.dlstcd <= 584)))
            delist['dlret'] = delist['dlret'].mask(ridx & ((delist.exchcd == 1) | (delist.exchcd == 2)), -0.35)
            delist['dlret'] = delist['dlret'].mask(ridx & (delist.exchcd == 3), -0.55)
        else:  # do nothing
            pass

        delist['dlret'] = delist['dlret'].astype(config.float_type)
        log(f'Missing dlret after filling: {delist.dlret.isna().sum()}')

        # Merge sf with delist
        sf = merge(sf, delist[['permno', 'date', 'dlret']], on=['date', 'permno'], how='left')

        is_ret_nan = sf.ret.isna() & sf.dlret.isna()
        sf.ret = (1 + sf.ret.fillna(0)) * (1 + sf.dlret.fillna(0)) - 1
        sf.loc[is_ret_nan, 'ret'] = np.nan  # Set ret = None if both ret and dlret missing.

        if not monthly:  # if dsf, stop here: no need to add cash dividend.
            elapsed_time(f'{f}sf and {f}seall merged. {f}sf shape: {sf.shape}')
            return sf

        #######################################
        # CASH DIST
        #######################################
        # Used in divi, divo.
        # Be careful there can be multiple distcd's for a (permno, date) pair.
        dist = seall[['permno', 'date', 'distcd', 'divamt']].dropna(subset=['distcd']).drop_duplicates()
        is_cash_div = (dist.distcd % 1000 // 100).isin([2, 3])  # cash if 2nd digit in (2, 3)
        dist['cash_div'] = np.where(is_cash_div, dist.divamt, np.nan)
        dist = dist.groupby(['date', 'permno'])['cash_div'].sum()
        sf = merge(sf, dist, on=['date', 'permno'], how='left')

        elapsed_time(f'{f}sf and {f}seall merged. {f}sf shape: {sf.shape}')
        return sf

    @staticmethod
    def add_gvkey_to_crsp(sf):
        """Add gvkey to m(d)sf and identify primary stocks.

        The permno and gvkey are mapped using crsp.ccmxpf_linktable.

        Primary stocks are identified in the following order.

            1. If linkprim = 'P' or 'C', set the security as primary.
            2. If permno and gvkey have 1:1 mapping, set the security as primary.
            3. Among the securities with the same gvkey, set the one with the maximum trading volume as primary.
            4. Among the securities with the same permco and missing gvkey, set the one with the maximum trading volume
               as primary.

        Args:
            sf: m(d)sf DataFrame with index = date/permno.

        Returns:
            m(d)sf with 'gveky' and 'primary' (primary stock indicator) columns added.

        References:
            https://wrds-www.wharton.upenn.edu/pages/support/research-wrds/macros/wrds-macros-cvccmlnksas/

            https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp-and-compustat/
        """

        elapsed_time('Adding gvkey to crsp...')

        link = WRDS.read_data('ccmxpf_linktable', 'crsp')
        link = link[~link.lpermno.isna()]
        link = link[link.usedflag == 1]
        link = link[link.linktype.isin(['LC', 'LU', 'LS'])]  # primary links only
        # link = link[link.linktype.isin(['LC', 'LN', 'LU', 'LX', 'LD', 'LS'])]  # primary and secondary links
        link = link[['lpermno', 'gvkey', 'linkdt', 'linkenddt', 'linkprim']].rename(columns={'lpermno': 'permno'})

        # Exceptions
        link.loc[(link.gvkey == '002759') & (link.linkdt == '1985-09-26') & (link.permno == 66843), 'linkprim'] = 'C'
        link.loc[(link.gvkey == '013699') & (link.linkdt == '1988-08-25') & (link.permno == 75189), 'linkprim'] = 'C'
        link.loc[(link.gvkey == '004162') & (link.linkdt == '2002-10-01') & (link.permno == 31318), 'linkprim'] = 'C'

        sf1 = sf[['permco']].reset_index()
        sf1['dvol'] = (sf.prc.abs() * sf.vol).to_numpy()  # dollar volume

        # To use merge_asof, data should be sorted.
        sf1 = sf1.sort_values('date')
        link = link.sort_values('linkdt')

        # merge sf with link. linkdt <= date <= linkenddt
        sf_idx = sf1.index
        sf1 = pd.merge_asof(sf1, link, left_on='date', right_on='linkdt', by='permno')
        sf1.set_index(sf_idx, inplace=True)
        sf1.loc[sf1.date > sf1.linkenddt, ['gvkey', 'linkprim']] = None

        # Primary identification
        sf1['primary'] = False
        sf1.loc[sf1.linkprim.isin(['P', 'C']), 'primary'] = True

        gb1 = sf1.groupby(['date', 'gvkey'])
        sf1.loc[gb1.permno.transform('size') == 1, 'primary'] = True  # Primary if permno and gveky are 1:1

        # Primary identification by volume
        # We first group by gvkey and then by permco since they are not always 1:1.
        sf1['max_dvol'] = gb1['dvol'].transform('max')
        sf1['nprimary'] = gb1.primary.transform('sum')  # num. primary (0 means unidentified)
        sf1.loc[(sf1.nprimary == 0) & (sf1.dvol == sf1.max_dvol), 'primary'] = True

        gb2 = sf1.groupby(['date', 'permco'])
        sf1['max_dvol'] = gb2['dvol'].transform('max')
        sf1['nprimary'] = gb2.primary.transform('sum')  # num. primary (0 means unidentified)
        sf1.loc[sf1.gvkey.isna() & (sf1.nprimary == 0) & (sf1.dvol == sf1.max_dvol), 'primary'] = True

        # Check missing or dup
        nprimary = gb1.primary.sum()
        if (nprimary != 1).any():
            log('Primary missing or dup.')
            log(nprimary[nprimary != 1])

        # sf1.set_index(['date', 'permno'], inplace=True)
        # sf[['gvkey', 'primary']] = sf1[['gvkey', 'primary']]
        sf1.sort_index(inplace=True)
        sf['gvkey'] = sf1['gvkey'].to_numpy()
        sf['primary'] = sf1['primary'].to_numpy()

        elapsed_time(f'gvkey added to crsp: crsp shape: {sf.shape}, missing gvkey: {sf.gvkey.isna().sum()}')
        log(f'No. unique permnos: {len(sf.index.get_level_values(-1).unique())}, no. unique gvkeys: {len(sf.gvkey.unique())}')
        return sf

    @staticmethod
    def create_crsp_comp_linktable():
        """Create a CRSP-Compustat link table using cusip.

        This method creates a CRSP-Compustat link table by merging crsp.msf with comp.security on cusip.
        This method can be used if the user does not have a WRDS subscription for ccmxpf_linktable.
        The link table has the columns ['cusip', 'gvkey', 'permno', 'linkdt', 'linkenddt', 'linkprim'] and is saved to
        ``config.input_dir/crsp_comp_linktable``. The linkprim column value is True if a security is primary.

        Note:
            The sql in the reference uses historical cusip (ncusip in msenames). However, we use cusip in msf as it
            renders more matches.

            A security is considered primary (linkprim = True) if its cusip is in funda or fundq.

        References:
            https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp-and-compustat/
        """

        elapsed_time('Creating crsp_comp_linktable...')

        # Load crsp.msf.
        crsp = WRDS.read_data('msf', 'crsp')
        crsp = crsp[['cusip']].reset_index()
        crsp = crsp.groupby('permno').agg({'cusip': 'first', 'date': ['min', 'max']})
        crsp = crsp.droplevel(1, axis=1)
        crsp.columns = ['cusip', 'linkdt', 'linkenddt']
        crsp = crsp.reset_index()
        # crsp['ncusip'] = crsp.groupby('permno', group_keys=False)['ncusip'].apply(lambda x: x.bfill().ffill())
        # crsp = crsp[['ncusip']].reset_index()
        # crsp = crsp.groupby(['permno', 'ncusip']).agg({'date': ['min', 'max']})
        # crsp = crsp.droplevel(1, axis=1).reset_index()
        # crsp.columns = ['permno', 'cusip', 'linkdt', 'linkenddt']

        # Load comp.security.
        sec = WRDS.read_data('security', 'comp')
        sec = sec[['cusip', 'gvkey']]
        sec['cusip'] = sec['cusip'].str[:8]
        sec = sec.drop_duplicates()

        # Merge msf with security on cusip.
        linktable = merge(crsp, sec, on='cusip', how='inner')

        # Primary identification
        # Load funda and fundq.
        funda = WRDS.read_data('funda', 'comp')
        fundq = WRDS.read_data('fundq', 'comp')
        comp = pd.concat([funda['cusip'], fundq['cusip']])
        del funda, fundq

        comp = comp.str[:8]
        comp = pd.DataFrame({'cusip': comp.unique()})
        comp['linkprim'] = True

        linktable = merge(linktable, comp, on='cusip', how='left')
        linktable = linktable[['cusip', 'gvkey', 'permno', 'linkdt', 'linkenddt', 'linkprim']]

        WRDS.save_data(linktable, 'crsp_comp_linktable')

        elapsed_time('crsp_comp_linktable created.')

    @staticmethod
    def add_gvkey_to_crsp_cusip(sf):
        """Add gvkey to m(d)sf and identify primary stocks using internal link table.

        The permno and gvkey are mapped using crsp_comp_linktable.

        Primary stocks are identified in the following order.

            1. If linkprim = True, set the security as primary.
            2. If permno and gvkey have 1:1 mapping, set the security as primary.
            3. Among the securities with the same gvkey, set the one with the maximum trading volume as primary.
            4. Among the securities with the same permco and missing gvkey, set the one with the maximum trading volume
               as primary.

        Args:
            sf: m(d)sf DataFrame with index = date/permno.

        Note:
            Compared to using ccmxpf_linktable, about 13% of gvkey's and 3% of primary's are different.

        Returns:
            m(d)sf with 'gveky' and 'primary' (primary stock indicator) columns added.
        """

        elapsed_time('Adding gvkey to crsp...')

        link = WRDS.read_data('crsp_comp_linktable')

        sf1 = sf[['permco']].reset_index()
        sf1['dvol'] = (sf.prc.abs() * sf.vol).to_numpy()  # dollar volume

        # # To use merge_asof, data should be sorted.
        # sf1 = sf1.sort_values('date')
        # link = link.sort_values('linkdt')
        #
        # # merge sf with link. linkdt <= date <= linkenddt
        # sf_idx = sf1.index
        # sf1 = pd.merge_asof(sf1, link, left_on='date', right_on='linkdt', by='permno')
        # sf1.set_index(sf_idx, inplace=True)
        # sf1.loc[sf1.date > sf1.linkenddt, 'gvkey'] = None

        sf1 = merge(sf1, link, on='permno', how='left')

        # Primary identification
        sf1['primary'] = False
        sf1.loc[sf1.linkprim == True, 'primary'] = True

        gb1 = sf1.groupby(['date', 'gvkey'])
        sf1.loc[gb1.permno.transform('size') == 1, 'primary'] = True  # Primary if permno and gveky are 1:1

        # Primary identification by volume
        # We first group by gvkey and then by permco since they are not always 1:1.
        sf1['max_dvol'] = gb1['dvol'].transform('max')
        sf1['nprimary'] = gb1.primary.transform('sum')  # num. primary (0 means unidentified)
        sf1.loc[(sf1.nprimary == 0) & (sf1.dvol == sf1.max_dvol), 'primary'] = True

        gb2 = sf1.groupby(['date', 'permco'])
        sf1['max_dvol'] = gb2['dvol'].transform('max')
        sf1['nprimary'] = gb2.primary.transform('sum')  # num. primary (0 means unidentified)
        sf1.loc[sf1.gvkey.isna() & (sf1.nprimary == 0) & (sf1.dvol == sf1.max_dvol), 'primary'] = True

        # Check missing or dup
        nprimary = gb1.primary.sum()
        if (nprimary != 1).any():
            log('Primary missing or dup.')
            log(nprimary[nprimary != 1])

        # sf1.sort_index(inplace=True)
        sf['gvkey'] = sf1['gvkey'].to_numpy()
        sf['primary'] = sf1['primary'].to_numpy()

        elapsed_time(f'gvkey added to crsp: crsp shape: {sf.shape}, missing gvkey: {sf.gvkey.isna().sum()}')
        log(f'No. unique permnos: {len(sf.index.get_level_values(-1).unique())}, no. unique gvkeys: {len(sf.gvkey.unique())}')
        return sf

    @staticmethod
    def preprocess_crsp(use_ccmxpf_linktable=None):
        """Create crspm and crspd files.

        This method calls :py:meth:`merge_sf_with_seall` and :meth:`add_gvkey_to_crsp` to add delist return,
        gveky, and primary indicator to m(d)sf. The result is saved to ``config.input_dir/crspm(d)``.

        Args:
            use_ccmxpf_linktable: If True, use crsp.ccmxpf_linktable to link CRSP and Compustat; if False, use
                internally created link table, crsp_comp_linktable. If None, use crsp.ccmxpf_linktable if it exists,
                otherwise, use crsp_comp_linktable.
        """

        if use_ccmxpf_linktable is None:
            if os.path.exists(config.input_dir + 'crsp/ccmxpf_linktable.' + config.file_format):
                use_ccmxpf_linktable = True
            else:
                use_ccmxpf_linktable = False

        if use_ccmxpf_linktable:
            add_gvkey_to_crsp = WRDS.add_gvkey_to_crsp
        else:
            add_gvkey_to_crsp = WRDS.add_gvkey_to_crsp_cusip

        crsp = WRDS.merge_sf_with_seall(monthly=True)
        crsp = add_gvkey_to_crsp(crsp)
        WRDS.save_data(crsp, 'crspm')

        crsp = WRDS.merge_sf_with_seall(monthly=False)
        crsp = add_gvkey_to_crsp(crsp)
        WRDS.save_data(crsp, 'crspd')

    @staticmethod
    def get_risk_free_rate(sdate=None, edate=None, src='mcti', month_end=False):
        """Get risk-free rate.

        The risk-free rate can be obtained either from crsp.mcti or ff.factors_monthly.
        The mcti is preferred since the values in factors_monthly have only 4 decimal places.
        Both risk-free rates are in decimal (not percentage values).

        Args:
            sdate: Start date.
            edate: End date.
            src: data source. 'mcti': crsp.mcti, 'ff': ff.factors_monthly.
            month_end: If True, shift dates to the end of the month.

        Returns:
            DataFrame of risk-free rates with index = 'date' and columns = ['rf'].
        """

        if src == 'mcti':
            rf = WRDS.read_data('mcti')[['caldt', 't30ret']]
            rf = rf.rename(columns={'caldt': 'date', 't30ret': 'rf'}).set_index('date')
        elif src == 'ff':
            rf = WRDS.read_data('factors_monthly')[['dateff', 'rf']]
            rf = rf.rename(columns={'dateff': 'date'}).set_index('date')
        else:
            raise ValueError(f'Undefined risk free data source: {src}')

        if month_end:
            rf.index = to_month_end(rf.index)
        return rf.loc[sdate:edate]

    @staticmethod
    def convert_fund_currency_to_usd(fund, table='funda'):
        """Convert non-USD values of funda(q) to USD values.

        Args:
            fund: funda(q) DataFrame with index = datadate/gvkey.
            table: 'funda' or 'fundq': indicator whether `fund` is funda or fundq.

        Returns:
            Converted `fund` DataFrame.

        NOTE:
            In Compustat North America, the accounting data can be either in USD and CAD. This is no problem if firm
            characteristics are generated using only Compustat. However, if data from different sources are mixed, e.g.,
            if CRSP's market equity (in USD) is combined with Compustat, Compustat data should be converted to USD.

            Following JKP, we use compustat.exrt_dly to obtain exchange rates. The exrt_dly starts from 1982-02-01.
        """

        curcol = 'curcd' if table == 'funda' else 'curcdq'

        exrt = WRDS.read_data('exrt_dly')
        usd_gbp = exrt.loc[exrt.tocurd == 'USD', ['datadate', 'exratd']]
        exrt = exrt.merge(usd_gbp, on='datadate', how='left')
        exrt['exratd_y'] /= exrt['exratd_x']
        exrt.rename(columns={'tocurd': curcol, 'exratd_y': 'exratd'}, inplace=True)

        fund = merge(fund, exrt[['datadate', curcol, 'exratd']], on=['datadate', curcol], how='left')

        not_usd = fund[curcol] != 'USD'
        exratd = fund.loc[not_usd, 'exratd'].values
        del fund['exratd']

        for col in fund:
            if is_float(fund[col]) and (col not in ['fyear', 'fyr', 'fyearq', 'fqtr']):
                fund.loc[not_usd, col] = fund.loc[not_usd, col].values * exratd

        return fund

    @staticmethod
    def save_data(data, table, library=None, index_col=None, sort_col=None, typecast=True):
        """Save downloaded table to a file.

        The file format can be either pickle (default) or parquet and configured by :func:`~.config.set_config`.
        The `data` is saved in the following location:

            * If `library` = None, ``config.input_dir/table``.
            * Otherwise, ``config.input_dir/library/table``.

        Args:
            data: Data to save (DataFrame).
            table: File name without extension.
            library: Directory.
            index_col: (List of) column(s) to be set as index.
            sort_col: (List of) column(s) to sort data on.
            typecast: If True, cast float to `config.float_type` and object to string before saving to a file.

        Note:
            A parquet file size can be significantly smaller especially when there are many duplicate values in columns.
            However, it tends to be slower to read and write and takes significantly more memory in some cases for
            unknown reasons. To change the file format to parquet, use ``set_config(file_format='parquet')``.
            We use parquet or pickle file format to store data as they preserve data types and are much faster to read
            compared to a csv file. To convert a file to a csv file, use :meth:`save_as_csv`.
        """

        fdir = config.input_dir + (library + '/' if library else '')

        if not os.path.isdir(fdir):
            os.makedirs(fdir)

        if sort_col is not None:
            data.sort_values(sort_col, inplace=True)
        if index_col is not None:
            data.set_index(index_col, inplace=True)

        if typecast:
            for col in data:
                if is_float(data[col]) and (data[col].dtype != config.float_type):
                    data[col] = data[col].astype(config.float_type)
                elif data[col].dtype == 'object':
                    if is_bool_array(data[col]):
                        data[col] = data[col].fillna(False).astype(bool)
                    else:
                        data[col] = data[col].astype('string')

        if config.file_format == 'parquet':
           data.to_parquet(fdir + table + '.parquet')
        else:
           data.to_pickle(fdir + table + '.pickle')

    @staticmethod
    def read_data(table, library=None, index_col=None, sort_col=None, typecast=True):
        """Read data from a saved table.

        The file path is ``config.input_dir/library/table``.
        The `library` argument is redundant: if it is None, all folders under ``config.input_dir`` is searched.

        Args:
            table: File name without extension.
            library: Directory.
            index_col: (List of) column(s) to be set as index.
            sort_col: (List of) column(s) to sort data on.
            typecast: If True, cast float to ``config.float_type`` and object to string after reading from the file.

        Returns:
            DataFrame. Data read. Index = `index_col`.
        """

        read_file = pd.read_parquet if config.file_format == 'parquet' else pd.read_pickle

        if library:
            fpath = config.input_dir + library + '/' + table + '.' + config.file_format
            data = read_file(fpath)
        else:  # if library is None, search all subdirectories.
            data = None
            for dir in os.walk(config.input_dir):
                fpath = dir[0] + '/' + table + '.' + config.file_format
                if os.path.exists(fpath):
                    data = read_file(fpath)
                    break

            if data is None:
                raise ValueError(f'[{table}] does not exist.')

        if (data.index.names[-1] == 'permno') and (data.index.dtypes.iloc[-1] == 'int64'):
            data.index = data.index.set_levels(data.index.levels[-1].astype('Int64'), level=-1)

        if sort_col is not None:
            data.sort_values(sort_col, inplace=True)
        if index_col is not None:
            if to_list(index_col) != data.index.names:
                if data.index.names[0]:
                    data.reset_index(inplace=True)
                data.set_index(index_col, inplace=True)

        if typecast:
            for col in data:
                if is_float(data[col]) and (data[col].dtype != config.float_type):
                    data[col] = data[col].astype(config.float_type)
                elif data[col].dtype == 'object':
                    if is_bool_array(data[col]):
                        data[col] = data[col].fillna(False).astype(bool)
                    else:
                        data[col] = data[col].astype('string')

        return data

    @staticmethod
    def save_as_csv(table, library=None, fpath=None):
        """Read a file and save it to a csv file.

        Args:
            table: File name without extension.
            library: Directory.
            fpath: File path for the csv file. If None, the file is saved to ``config.input_dir/library/table.csv``.
        """

        fpath = fpath or config.input_dir + (library + '/' if library else '') + table + '.csv'
        data = WRDS.read_data(table, library)
        data.to_csv(fpath)


    ################################################################################
    #
    # UNDER CONSTRUCTION
    #
    ################################################################################
    # def download_ibes(self):
    #     log('This function is under construction.')
    #     return
    #
    #     log('downloading ibes...')
    #     sql = """
    #         SELECT
    #            ticker, cusip, fpedats, statpers, anndats_act, numest, anntims_act, medest, actual, stdev
    #         FROM
    #             ibes.statsum_epsus
    #         /*WHERE fpi='6'   1 is for annual forecasts, 6 is for quarterly */
    #         WHERE statpers < anndats_act  /* only keep summarized forecasts prior to earnings annoucement */
    #         AND measure = 'EPS'
    #         AND medest is not null
    #         AND fpedats is not null
    #         AND fpedats >= statpers
    #         ORDER BY cusip, fpedats, statpers
    #         """
    #
    #     data = self.db.raw_sql(sql)
    #     data.fpedats = pd.to_datetime(data.fpedats)
    #     save_data(data, 'ibes')
    #     log(f'download complete.')
    #
    # def add_permno_to_fund(self, fd, dropna=True):
    #     log('This function is under construction.')
    #     return
    #
    #     if 'permno' in fd.columns:
    #         return fd
    #
    #     link = read_data('ccmxpf_linktable', 'crsp')
    #     link = link[~link.lpermno.isna()]
    #     link = link[link.usedflag == 1]
    #     link = link[link.linktype.isin(['LC', 'LN', 'LU', 'LX', 'LD', 'LS'])]
    #     link = link[['lpermno', 'gvkey', 'linkdt', 'linkenddt']].rename(columns={'lpermno': 'permno'})
    #
    #     link.linkdt = pd.to_datetime(link.linkdt)
    #     link.linkenddt = pd.to_datetime(link.linkenddt)
    #
    #     fd = fd.sort_index().reset_index()
    #     link = link.sort_values(['linkdt', 'gvkey'])
    #
    #     fd = pd.merge_asof(fd, link, left_on='datadate', right_on='linkdt', by='gvkey')
    #     # fd = fd[(fd.datadate <= fd.linkenddt) | fd.linkenddt.isna()]
    #     fd.loc[fd.datadate > fd.linkenddt, 'permno'] = None
    #
    #     fd.drop(columns=['linkdt', 'linkenddt'], inplace=True)
    #
    #     fd = fd.sort_values(['gvkey', 'datadate']).set_index(['datadate', 'gvkey'])
    #     if dropna:
    #         log(f'fd size: {len(fd)}')
    #         fd = fd[~fd.permno.isna()]
    #         log(f'fd size with not missing permno: {len(fd)}')
    #     return fd
    #
    # def merge_ibes_crspm(self, ibes, crspm):
    #     """
    #     https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-ibes-and-crsp-data/
    #     * Merging IBES and CRSP datasets using ICLINK table;
    #     proc sql;
    #     create table IBES_CRSP
    #     as select a.ticker, a.STATPERS, a.meanrec, c.permno, c.date, c.ret from ibes.recdsum as a,
    #     home.ICLINK as b,
    #     crsp.msf as c
    #     where a.ticker=b.ticker and b.permno=c.permno and intnx('month',a.STATPERS,0,'E') = intnx('month',c.date,0,'E');
    #     quit;
    #     """
    #     log('This function is under construction.')
    #     return
    #
    #     db = sqlite3.connect(':memory:')
    #
    #     link = read_data('ibcrsphist', 'wrdsapp')
    #
    #     # write the tables
    #     ibes.reset_index()[['cusip', 'fpedats']].to_sql('ibes', db, index=False)
    #     link.to_sql('link', db, index=False)
    #     crspm.reset_index()[['date', 'permno']].to_sql('crspm', db, index=False)
    #     elapsed_time('tables uploaded to db')
    #
    #     # Merge ibes with link
    #     sql = """
    #         CREATE TABLE temp as
    #         SELECT l.lpermno as permno, f.*
    #         FROM
    #             ibes as f, link as l
    #         /*ON f.cusip = l.lcusip
    #         AND f.fpedats >= sdate
    #         AND f.fpedats <= edate
    #         WHERE l.lpermno is not null*/
    #         """
    #
    #     db.execute(sql)
    #     elapsed_time('funda and link merged')
    #
    #     sql = """
    #         SELECT c.*, f.*
    #         FROM
    #             crspm as c
    #         LEFT JOIN
    #             temp as f
    #         ON c.permno = f.permno
    #         AND c.date >= f.datadate + 7
    #         AND c.date < f.datadate + 20
    #         """
    #
    #     table = pd.read_sql_query(sql, db)
    #     return table


if __name__ == '__main__':
    os.chdir('../')

    wrds = WRDS('')
    wrds.download_all()
    wrds.preprocess_crsp()
