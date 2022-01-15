import shutil
import pandas as pd
import datetime as dt
import sqlite3
import wrds
import asyncio

from pyanomaly.globals import *
from pyanomaly.datatools import to_month_end
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
#  5: Units (Depository Units, Units of Beneficial Interest, Units of Limited Partnership Interest, Depository Receipts, etc.)
#  2nd digit
#  0: Securities which have not been further defined.
#  1: Securities which need not be further defined.
#  2: Companies incorporated outside the US
#  3: Americus Trust Components (Primes and Scores).
#  4: Closed-end funds.
#  5: Closed-end fund companies incorporated outside the US
#  8: REIT's (Real Estate Investment Trusts).
################################################################################


class WRDS():
    """Class to download/handle WRDS data.

    Args:
        wrds_username: WRDS username. Necessary only when downloading data: can be set to None when reading
            data from files.

    **Methods for Data Download**

        * ``download_table()``
        * ``download_table_async()``
        * ``download_funda()``
        * ``download_fundq()``
        * ``download_sf()``
        * ``download_seall()``
        * ``download_secd()``
        * ``download_g_secd()``
        * ``download_all()``

    **Other Methods**

        * ``create_pgpass_file()``
        * ``merge_sf_with_seall()``
        * ``add_gvkey_to_crsp()``
        * ``preprocess_crsp()``
        * ``convert_fund_currency_to_usd()``
        * ``get_risk_free_rate()``
        * ``add_gvkey_to_crsp()``
        * ``save_data()``
        * ``read_data()``
        * ``save_as_csv()``

    References:
        CRSP overview: https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/

        CRSP-Compustat merge: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/crspcompustat-merged-ccm/wrds-overview-crspcompustat-merged-ccm/

        CRSP annual update tables: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_indexes/

        shrcd: https://wrds-www.wharton.upenn.edu/data-dictionary/form_metadata/crsp_a_stock_msf_identifyinginformation/shrcd/

        exchcd: https://wrds-www.wharton.upenn.edu/data-dictionary/form_metadata/crsp_a_stock_msf_identifyinginformation/exchcd/

    Attributes:
        db: A wrds object to connect to WRDS database.
    """
    def __init__(self, wrds_username=None):
        if wrds_username:
            self.db = wrds.Connection(wrds_username=wrds_username)

    def create_pgpass_file(self):
        """Create pgpass file. Need to be called only once. Once created, you don't need to enter password when
        connecting to WRDS.
        """

        self.db.create_pgpass_file()

    ################################################################################
    #
    # DATA DOWNLOAD
    #
    ################################################################################
    def download_table(self, library, table, obs=-1, offset=0, columns=None, coerce_float=None, index_col=None,
                       date_cols=None):
        """Download a table from WRDS library.

        This is a wrapping function of ``wrds.get_table()``.
        The queried table is saved in config.input_dir/library/table.pickle. Useful when downloading an entire table.

        Args:
            library: WRDS library. e.g., crsp, comp, ...
            table: A table in `library`.
            obs: See ``wrds.get_table()``.
            offset: See ``wrds.get_table()``.
            columns: See ``wrds.get_table()``.
            coerce_float: See ``wrds.get_table()``.
            index_col: See ``wrds.get_table()``.
            date_cols: See ``wrds.get_table()``.
        """

        elapsed_time(f'Downloading {library}.{table}...')
        data = self.db.get_table(library, table, obs, offset, columns, coerce_float, index_col, date_cols)
        data = self._convert_types(data, [(library, table)])

        self.save_data(data, table, library)
        elapsed_time(f'Download complete. Size: {data.shape}')

    def _convert_types(self, data, src_tables, type=float, date_cols=None):
        """Convert dtype of double precision fields to float.

        The dtype of a double precision field is object when first
        downloaded if the field contains None, which can significantly slower computations.

        Args:
            data: Dataframe of which dtypes to be converted.
            src_tables: list of (library, table) that are used to create the data.
            type: float or np.float32. Numeric fields will be converted to this type. For a large dataset, converting
            to float32 can save disc space and read/write time.
            date_cols: list of date columns. The dtype of these columns will be converted to pd.Timestamp.

        Returns:
            data: type-converted data.
        """

        desc = [self.db.describe_table(*src) for src in src_tables]  # get table descriptions
        desc = pd.concat(desc).drop_duplicates(subset=['name'])

        for col in data:
            if str(desc.loc[desc.name == col, 'type'].values[0]) == 'DOUBLE PRECISION':
                data[col] = data[col].astype(type, errors='ignore')
            if (date_cols is not None) and (col in date_cols):
                data[col] = pd.to_datetime(data[col])

        return data

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
        if data.empty:
            return

        self.save_data(data, f"{table}_{sdate.replace('-', '')}_{edate.replace('-', '')}", 'tmp')
        log(f"Download complete. {table}: {sdate}-{edate}")

    def download_table_async(self, library, table, sql=None, date_col=None, sdate=None, edate=None, interval=5, src_tables=None, type=float, date_cols=None, run_in_executer=True):
        """Asynchronous download.

        This method splits the total period into `interval` years and downloads data for each sub-period
        asynchronously. If download fails, it can be started from the failed date: already downloaded files will be
        gathered together. This method allow us to download a large table, e.g., crsp.dsf, reliably without connection
        timeout and consumes much less memory than ``WRDS.download_table()``.

        NOTE:
            For a small table, this can be slower than ``WRDS.download_table()``.
            Table should have a date field (`date_col`) to split the period.

        Args:
            library: WRDS library.
            table: WRDS table. If a complete query is given, this can be any name: this is used only as the file name
                for the data.
            sql: String of the fields to select or a complete query statement. See below.
            date_col: Date field on which downloaing will be split.
            sdate: Start date ('yyyy-mm-dd'). If None, '1900-01-01'.
            edate: End date. If None, today.
            interval: Sub-period size in years.
            src_tables: List of (library, table) that are used in the query. `src_tables` are used to get data types
                and convert the types of double precision fields to float. When data is selected from a single table,
                `library.table`, this can be set to None.
            type: Type for numeric fields: float or np.float32. For a large dataset, converting to float32 can save
                disc space and read/write time.
            date_cols: List of date columns. The dtype of these columns will be converted to datetime.
            run_in_executer: If True, download data concurrently using `executer`. Setting this to True will increase
                download speed but can take up much memory.

        NOTE:
            * To download library.table, `library`, `table`, and `date_col` should be given.
            * To download selected fields from library.table, library, `table`, `sql`, and `date_col` should be given, where
              `sql` is a string of the fields to select, e.g.,

              >>> wrds = WRDS('user_name')
              >>> sql = 'permno, prc, exchcd'
              >>> wrds.download_table_async('crsp', 'msf', sql, 'date')

            * To download data using a complete query, `library`, `table`, and `sql` should be given, where `table` is a table
              name for the data (can be any name), and `sql` is a query statement. The `sql` should contain
              'WHERE [`date_col`] BETWEEN {} and {}'. See, e.g., the code of ``WRDS.download_sf()``.
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
            if run_in_executer:
                loop = asyncio.get_event_loop()
                # aws = [self._download_table_async(table, sql, sdate, edate) for sdate, edate in dates]
                aws = [loop.run_in_executor(None, self._download_table_async, table, sql, sdate, edate) for sdate, edate in dates]
                loop.run_until_complete(asyncio.gather(*aws))
            else:
                for sdate, edate in dates:
                    self._download_table_async(table, sql, sdate, edate)


            # Read saved data files and save them as one big file.
            log('Gathering files...')
            tmp_dir = config.input_dir + 'tmp'  # sub-period files are temporarily saved here.
            data = []
            for f in os.listdir(tmp_dir):
                fpath = os.path.join(tmp_dir, f)
                if os.path.isfile(fpath) and (f[:len(table)] == table):
                    data.append(pd.read_pickle(fpath))
            data = pd.concat(data)

            for file in os.scandir(tmp_dir):
                print(file)
                os.remove(file.path)

        # type conversion
        if src_tables is None:
            src_tables = [(library, table)]
        data = self._convert_types(data, src_tables, type, date_cols)

        self.save_data(data, table, library)
        elapsed_time(f'Download complete: {table}')

    def download_sf(self, sdate=None, edate=None, monthly=True, run_in_executer=True):
        """ Download crsp.m(d)sf joined with crsp.m(d)senames.

        Args:
            sdate: Start date. e.g., '2000-01-01'. Set to None to download all data.
            edate: End date. Set to None to download all data.
            monthly: If True download msf else dsf.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """

        if monthly:
            fields = 'a.*, b.shrcd, b.exchcd, b.siccd'
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
        type = float if monthly else np.float32  # For dsf, use float32 to save space.
        self.download_table_async('crsp', sf, sql, sdate=sdate, edate=edate, src_tables=src_tables, type=type, date_cols=['date'], run_in_executer=run_in_executer)

    def download_seall(self, sdate=None, edate=None, monthly=True, run_in_executer=True):
        """Download delist info from m(d)seall.

        Delist can be obtained from either mseall or msedelist. We use mseall since it contains exchcd, which is used
        when replacing missing dlret.
        The shrcd and exchcd in mseall are usually those before halting/suspension. If a stock in NYSE is halted, exchcd
        in msenames can be -2, whereas that in mseall is 1.

        Args:
            sdate: Start date. e.g., '2000-01-01'. Set to None to download all data.
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

        type = float if monthly else np.float32
        self.download_table_async('crsp', seall, sql, sdate=sdate, edate=edate, type=type, date_cols=['date'], run_in_executer=run_in_executer)

    def download_funda(self, sdate=None, edate=None, run_in_executer=True):
        """Download comp.funda.

        Args:
            sdate: Start date. e.g., '2000-01-01'. Set to None to download all data.
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
        self.download_table_async('comp', 'funda', sql, sdate=sdate, edate=edate, src_tables=src_tables, date_cols=['datadate'], run_in_executer=run_in_executer)

    def download_fundq(self, sdate=None, edate=None, run_in_executer=True):
        """Download comp.fundq.

        Args:
            sdate: Start date. e.g., '2000-01-01'. Set to None to download all data.
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
                ceqq, cshoq, prccq, rdq, pstkrq, aqaq, aqdq, aqepsq, aqpq, arcedq,
                arceepsq, arceq, cibegniq, cicurrq, ciderglq, 
                oibdpq, oiadpq, miiq, xoprq, xiq, doq, xidoq,
                ivaoq, ivstq, req, txditcq, txdbq, seqq, mibq, icaptq, ajexq, cshprq, epspxq, 
                saley, revty, cogsy, xsgay,
                dpy, xrdy, iby, nopiy, spiy, piy, niy, txty, xinty, capxy, oancfy, gdwliay, gdwlipy,
                txdiy, scstkcy, oibdpy, oiadpy, miiy, xopry, xiy, doy, xidoy, ibcy, dpcy, xidocy, 
                fincfy, fiaoy, txbcofy, dltry, dlcchy, prstkcy, sstky, dvy, dltisy
                afudciy, amcy, aolochy, apalchy, aqay, aqcy, aqdy, aqepsy, aqpy, arcedy, 
                arceepsy, arcey, cdvcy, chechy, cibegniy, cicurry, cidergly, wcapq, 
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
        self.download_table_async('comp', 'fundq', sql, sdate=sdate, edate=edate, src_tables=src_tables, date_cols=['datadate'], run_in_executer=run_in_executer)

    def download_secd(self, sdate=None, edate=None, run_in_executer=True):
        """Download comp.secd.

        Args:
            sdate: Start date. e.g., '2000-01-01'. Set to None to download all data.
            edate: End date. Set to None to download all data.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """
        sql = 'gvkey, datadate, prccd, ajexdi, cshoc, iid'
        self.download_table_async('comp', 'secd', sql, 'datadate', sdate, edate, run_in_executer=run_in_executer)

    def download_g_secd(self, sdate=None, edate=None, run_in_executer=True):
        """Download comp.g_secd.

        Args:
            sdate: Start date. e.g., '2000-01-01'. Set to None to download all data.
            edate: End date. Set to None to download all data.
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """
        sql = 'gvkey, datadate, prccd, ajexdi, cshoc, iid'
        self.download_table_async('comp', 'g_secd', sql, 'datadate', sdate, edate, run_in_executer=run_in_executer)

    def download_all(self, run_in_executer=True):
        """Download all tables.

        Currently, this function downloads the following tables:

        * comp.funda
        * comp.fundq
        * crsp.msf (merged with crsp.msenames)
        * crsp.dsf (merged with crsp.dsenames)
        * crsp.mseall
        * crsp.dseall
        * crsp.ccmxpf_linktable
        * crsp.mcti
        * ff.factors_monthly
        * ff.factors_daily
        * comp.exrt_dly

        Args:
            run_in_executer: If True, download concurrently. Faster but memory hungrier.
        """

        self.download_funda(run_in_executer=run_in_executer)
        self.download_fundq(run_in_executer=run_in_executer)
        self.download_sf(monthly=True, run_in_executer=run_in_executer)
        self.download_sf(monthly=False, run_in_executer=run_in_executer)
        self.download_seall(monthly=True, run_in_executer=run_in_executer)
        self.download_seall(monthly=False, run_in_executer=run_in_executer)

        self.download_table('crsp', 'ccmxpf_linktable', date_cols=['linkdt', 'linkenddt'])
        self.download_table('crsp', 'mcti', date_cols=['caldt'])
        self.download_table('ff', 'factors_monthly', date_cols=['date', 'dateff'])
        self.download_table('ff', 'factors_daily', date_cols=['date'])
        self.download_table('comp', 'exrt_dly', date_cols=['datadate'])

    ################################################################################
    #
    # UTILITIES
    #
    ################################################################################
    @staticmethod
    def merge_sf_with_seall(sf, monthly=True, fill_method=1):
        """Merge m(d)sf with m(d)seall.

        This method adjusts m(d)sf return ('ret') using m(d)seall delist return ('dlret'). The adjusted return replaces
        'ret' and 'dlret' column is added to m(d)sf.
        For msf, this method also adds cash dividend columns, 'cash_div', to m(d)sf.

        Args:
            sf: m(d)sf DataFrame.
            monthly: `sf` = msf if True, else `sf` = dsf.
            fill_method: Method to fill missing dlret. 0: don't fill, 1: JKP code, or 2: GHZ code.

                - fill_method = 1:

                    - `dlret` = -0.30 if `dlstcd` is 500 or between 520 and 584.
                - fill_method = 2:

                    - `dlret` = -0.35 if `dlstcd` is 500 or between 520 and 584, and `exchcd` is 1 or 2.
                    - `dlret` = -0.55 if `dlstcd` is 500 or between 520 and 584, and `exchcd` is 3.

        Returns:
            m(d)sf with adjusted return (and cash dividend).

        References:
            Delist codes: http://www.crsp.com/products/documentation/delisting-codes

        """

        f = 'm' if monthly else 'd'
        elapsed_time(f'Merging {f}sf with {f}seall...')

        seall = WRDS.read_data(f'{f}seall')

        #######################################
        # DLRET
        #######################################
        delist = seall[['permno', 'date', 'dlret', 'dlstcd', 'shrcd', 'exchcd']].dropna(subset=['dlstcd']).drop_duplicates().copy()
        log(f'{f}sf shape: {sf.shape}, delist{f} shape: {delist.shape}')

        # Fill missing delist return
        log(f'Missing dlret: {delist.dlret.isna().sum()}')
        if fill_method == 1:  # from JKP SAS code
            ridx = delist.dlret.isnull() & ((delist.dlstcd == 500) | ((delist.dlstcd >= 520) & (delist.dlstcd <= 584)))
            delist.loc[ridx, 'dlret'] = -0.30
        elif fill_method == 2:  # from GHZ SAS code
            ridx = delist.dlret.isnull() & ((delist.dlstcd == 500) | ((delist.dlstcd >= 520) & (delist.dlstcd <= 584)))
            delist.loc[ridx & ((delist.exchcd == 1) | (delist.exchcd == 2)), 'dlret'] = -0.35
            delist.loc[ridx & (delist.exchcd == 3), 'dlret'] = -0.55
        else:  # do nothing
            pass

        log(f'Missing dlret after filling: {delist.dlret.isna().sum()}')

        # Merge sf with delist
        sf = pd.merge(sf, delist[['permno', 'date', 'dlret']], on=['permno', 'date'], how='left')

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
        dist = seall[['permno', 'date', 'distcd', 'divamt']].dropna(subset=['distcd']).drop_duplicates().copy()
        dist['is_cash_div'] = (dist.distcd % 1000 // 100).isin([2, 3])  # cash if 2nd digit in (2, 3)
        dist.loc[dist.is_cash_div, 'cash_div'] = dist.loc[dist.is_cash_div, 'divamt']
        dist = dist.groupby(['permno', 'date'])['cash_div'].sum()
        # if not monthly:
        #     dist['cash_div'] = dist['cash_div'].astype('float32')
        sf = pd.merge(sf, dist, on=['permno', 'date'], how='left')

        elapsed_time(f'{f}sf and {f}seall merged. {f}sf shape: {sf.shape}')
        return sf

    @staticmethod
    def add_gvkey_to_crsp(sf):
        """Add gvkey to m(d)sf using ccmxpf_linktable and identify primary stocks.

        There are two tables we can use to link permno with gvkey: ccmxpf_linktable and ccmxpf_linkhist.
        ccmxpf_lnkused is simply a merge table of ccmxpf_lnkhist and ccmxpf_lnkused.

        We identify primary stocks in the following order.

            1. If linkprim = 'P' or 'C', set the security as primary.
            2. If permno and gvkey have 1:1 mapping, set the security as primary.
            3. Among the securities with the same gvkey, set the one with the maximum trading volume as primary.
            4. Among the securities with the same permco and missing gvkey, set the one with the maximum trading volume
               as primary.

        References:
            https://wrds-www.wharton.upenn.edu/pages/support/research-wrds/macros/wrds-macros-cvccmlnksas/

            https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp-and-compustat/

        Args:
            sf: m(d)sf Dataframe with index = 'date'/'permno'.

        Returns:
            m(d)sf with 'gveky' and 'primary' (primary stock indicator) columns added.
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

        # To use merge_asof, data should be sorted.
        sf.sort_values('date', inplace=True)
        link.sort_values('linkdt', inplace=True)

        # merge sf with link. linkdt <= date <= linkenddt
        sf['permno'] = sf['permno'].astype('Int64')  # We need this for 'by=permno'.
        link['permno'] = link['permno'].astype('Int64')
        sf = pd.merge_asof(sf, link, left_on='date', right_on='linkdt', by='permno')
        sf.loc[sf.date > sf.linkenddt, ['gvkey', 'linkprim']] = None

        sf['primary'] = False
        sf.loc[sf.linkprim.isin(['P', 'C']), 'primary'] = True
        sf.loc[sf.groupby(['date', 'gvkey']).permno.transform('size') == 1, 'primary'] = True  # Primary if permno and gveky are 1:1

        # Primary identification by volume
        # We first group by gvkey and then by permco since they are not always 1:1.
        sf['dvol'] = sf.prc.abs() * sf.vol  # dollar volume
        sf['max_dvol'] = sf.groupby(['date', 'gvkey'])['dvol'].transform(max)
        sf['nprimary'] = sf.groupby(['date', 'gvkey']).primary.transform(sum)  # num. primary (0 means unidentified)
        sf.loc[(sf.nprimary == 0) & (sf.dvol == sf.max_dvol), 'primary'] = True

        sf['max_dvol'] = sf.groupby(['date', 'permco'])['dvol'].transform(max)
        sf['nprimary'] = sf.groupby(['date', 'permco']).primary.transform(sum)  # num. primary (0 means unidentified)
        sf.loc[sf.gvkey.isna() & (sf.nprimary == 0) & (sf.dvol == sf.max_dvol), 'primary'] = True

        # Check missing or dup
        nprimary = sf.groupby(['date', 'gvkey']).primary.sum()
        if (nprimary != 1).any():
            log('Primary missing or dup.')
            log(nprimary[nprimary != 1])

        sf.drop(columns=['linkdt', 'linkenddt', 'linkprim', 'dvol', 'max_dvol', 'nprimary'], inplace=True)
        sf.sort_values(['permno', 'date'], inplace=True)

        elapsed_time(f'gvkey added to crsp: crsp shape: {sf.shape}, missing gvkey: {sf.gvkey.isna().sum()}')
        log(f'No. unique permnos: {len(sf.permno.unique())}, no. unique gvkeys: {len(sf.gvkey.unique())}')
        return sf

    @staticmethod
    def preprocess_crsp():
        """Create crspm and crspd.

        This method calls ``WRDS.merge_sf_with_seall()`` and ``WRDS.add_gvkey_to_crsp()`` to add delist return,
        gveky, and primary indicator to m(d)sf.
        The result is saved to config.input_dir/crspm(d).pickle.
        """

        crsp = WRDS.read_data('msf')
        # msenames can be missing when a firm is delisted, resulting in missing shrcd/exchcd.
        # => fill with the latest data.
        crsp.sort_values(['permno', 'date'], inplace=True)
        crsp['shrcd'] = crsp.groupby('permno').shrcd.fillna(method='ffill')
        crsp['exchcd'] = crsp.groupby('permno').exchcd.fillna(method='ffill')

        crsp = WRDS.merge_sf_with_seall(crsp, monthly=True)
        crsp = WRDS.add_gvkey_to_crsp(crsp)
        WRDS.save_data(crsp, 'crspm')

        crsp = WRDS.read_data('dsf')
        crsp = WRDS.merge_sf_with_seall(crsp, monthly=False)
        crsp = WRDS.add_gvkey_to_crsp(crsp)
        WRDS.save_data(crsp, 'crspd')

    @staticmethod
    def get_risk_free_rate(sdate=None, edate=None, src='mcti', month_end=False):
        """Get risk-free rate.

        The risk free rate can be obtained either from crsp.mcti or ff.factors_monthly.
        mcti is preferred since the values in factors_monthly have only 4 decimal places.
        Both risk-free rates are in decimal (not percentage values).

        Args:
            sdate: Start date.
            edate: End date.
            src: data source. 'mcti': crsp.mcti, 'ff': ff.factors_monthly.
            month_end: Shift dates to the end of the month.

        Returns:
            Dataframe of risk-free rates with index = 'date' and columns = ['rf'].
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

        NOTE:
            In Compustat North America, the accounting data can be in either USD and CAD. This is no problem if firm
            characteristics are generated using only Compustat. However, if you mix data from different databases, e.g.,
            if you use market equity of CRSP, which is in USD, Compustat data should be converted to USD.

            Following JKP, we use compustat.exrt_dly to obtain exchange rates. exrt_dly starts from 1982-02-01.

        Args:
            fund: funda(q) DataFrame with index = 'datadate'/'gvkey'.
            table: 'funda' or 'fundq': indicator whether `fund` is funda or fundq.

        Returns:
            Converted `fund` DataFrame.
        """

        curcol = 'curcd' if table == 'funda' else 'curcdq'

        exrt = WRDS.read_data('exrt_dly')
        usd_gbp = exrt.loc[exrt.tocurd == 'USD', ['datadate', 'exratd']]
        exrt = exrt.merge(usd_gbp, on='datadate', how='left')
        exrt['exratd_y'] /= exrt['exratd_x']
        exrt.rename(columns={'tocurd': curcol, 'exratd_y': 'exratd'}, inplace=True)

        index_names = fund.index.names
        fund = fund.reset_index().merge(exrt[['datadate', curcol, 'exratd']], on=['datadate', curcol], how='left')
        fund.loc[fund[curcol] == 'USD', 'exratd'] = 1

        for col in fund:
            if (fund[col].dtype == float) and (col not in ['fyear', 'fyr', 'fyearq', 'fqtr']):
                fund[col] *= fund['exratd']

        return fund.set_index(index_names)

    @staticmethod
    def save_data(data, table, library=None):
        """Save `data` in pickle format.

        We use pickle file format to store data as a pickle file preserves data types and is much faster to read
        compared to a csv file. To convert a pickle file to a csv file, ``WRDS.save_as_csv()`` can be used.
        The `data` is saved in the following location:

            * If `library` = None, config.input_dir/`table`.pickle
            * Otherwise, config.input_dir/`library`/`table`.pickle

        Args:
            data: Data to save (DataFrame).
            table: File name without extension.
            library: Directory.
        """

        fdir = config.input_dir + (library + '/' if library else '')

        if not os.path.isdir(fdir):
            os.makedirs(fdir)

        data.to_pickle(fdir + table + '.pickle')

    @staticmethod
    def read_data(table, library=None, index_col=None, sort_col=None):
        """Read data from config.input_dir/`library`/`table`.pickle.

        `library` argument is redundant as if it is None, all folders under config.input_dir is searched.

        Args:
            table: File name without extension.
            library: Directory.
            index_col: (List of) column(s) to be set as index.
            sort_col: (List of) column(s) to sort data.

        Returns:
            (DataFrame) data read. Index = `index_col`.
        """

        data = None

        if library:
            fpath = config.input_dir + library + '/' + table + '.pickle'
            data = pd.read_pickle(fpath)
        else:  # if library is None, search all subdirectories.
            for dir in os.walk(config.input_dir):
                fpath = dir[0] + '/' + table + '.pickle'
                if os.path.exists(fpath):
                    data = pd.read_pickle(fpath)
        if data is None:
            raise ValueError(f'[{table}] does not exist.')
        if sort_col is not None:
            data.sort_values(sort_col, inplace=True)
        if index_col is not None:
            data.set_index(index_col, inplace=True)

        return data

    @staticmethod
    def save_as_csv(table, library=None, fpath=None):
        """Read `table` and save it to a csv file.

        Args:
            table: File name without extension.
            library: Directory.
            fpath: File path for the csv file. If None, the file is saved to config.input_dir/`library`/`table`.csv
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

    wrds = WRDS('fehouse')
    # wrds.download_all()
    # wrds.preprocess_crsp()
    wrds.download_table('ff', 'factors_monthly', date_cols=['date', 'dateff'])


