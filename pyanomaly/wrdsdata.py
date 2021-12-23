import shutil
import pandas as pd
import datetime as dt
import sqlite3
import wrds
import asyncio

from pyanomaly.globals import *
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
    def __init__(self, wrds_username=None):
        if wrds_username:
            self.db = wrds.Connection(wrds_username=wrds_username)

    ################################################################################
    #
    # DATA DOWNLOAD
    #
    ################################################################################
    def convert_types(self, data, src_tables, type=np.float):
        """Convert dtype of double precision fields to float. dtype of a double precision field is object when first
        downloaded if the field contains None, which can significantly slower computations.

        Args:
            data: Dataframe of which dtypes to be converted.
            src_tables: list of tuple, (library, table), that are used to create the data.

        Returns:
            data: type-converted data.
        """
        desc = [self.db.describe_table(*src) for src in src_tables]  # get table descriptions
        desc = pd.concat(desc).drop_duplicates(subset=['name'])

        for col in data:
            # if col in ('permno', 'lpermno', 'exchcd', 'shrcd'):
            #     data[col] = data[col].astype('Int32')
            if str(desc.loc[desc.name == col, 'type'].values[0]) == 'DOUBLE PRECISION':
                data[col] = data[col].astype(type, errors='ignore')

        return data

    def download_table(self, library, table, obs=-1, offset=0, columns=None, coerce_float=None, index_col=None,
                       date_cols=None):
        """Download table from WRDS library. The queried table is saved in config.rawdata_dir/library/table.pickle

        Args:
            library: WRDS library. e.g., crsp, comp, ...
            table: table in library.
            obs: see wrds.get_table()
            offset: see wrds.get_table()
            columns: see wrds.get_table()
            coerce_float: see wrds.get_table()
            index_col: see wrds.get_table()
            date_cols: see wrds.get_table()
        """
        elapsed_time(f'Downloading {library}.{table}...')
        data = self.db.get_table(library, table, obs, offset, columns, coerce_float, index_col, date_cols)
        data = self.convert_types(data, [(library, table)])

        self.save_data(data, table, library)
        elapsed_time(f'Download complete. Size: {data.shape}')

    async def download_sf(self, sdate, edate, monthly=True):
        """ Download m(d)sf joined with m(d)senames. Downloaded table is saved to crspm(d).pickle

        Args:
            sdate: start date. e.g., '2000-01-01'
            edate: end date
            monthly: If True download msf else dsf.
        """

        f = 'm' if monthly else 'd'
        log(f'Downloading {f}sf: {sdate}-{edate}')
        await asyncio.sleep(.01)

        if monthly:
            fields = 'a.*, b.shrcd, b.exchcd, b.siccd'
        else:  # for crspd, download only necessary columns to reduce file size.
            fields = 'a.permno, a.permco, a.date, bidlo, askhi, prc, ret, vol, shrout, cfacpr, cfacshr, b.shrcd, b.exchcd'

        sql = f"""
            SELECT {fields}
            FROM crsp.{f}sf as a
            LEFT JOIN crsp.{f}senames as b
            ON a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date <= b.nameendt
            WHERE a.date between '{sdate}' and '{edate}'
            ORDER BY a.permno, a.date
        """

        data = self.db.raw_sql(sql)
        if data.empty:
            return

        if monthly:
            data = self.convert_types(data, [('crsp', f'{f}sf'), ('crsp', f'{f}senames')])
        else:
            data = self.convert_types(data, [('crsp', f'{f}sf'), ('crsp', f'{f}senames')], np.float32)
        data.date = pd.to_datetime(data.date)

        # msenames can be missing if the firm is delisted, resulting in missing shrcd/exchcd.
        # => fill with the latest data.
        data.shrcd = data.shrcd.fillna(method='ffill')
        data.exchcd = data.exchcd.fillna(method='ffill')

        self.save_data(data, f"{f}sf_{sdate.replace('-', '')}_{edate.replace('-', '')}", 'tmp')
        log(f"Download complete: {data.date.min().strftime('%Y-%m-%d')}-{data.date.max().strftime('%Y-%m-%d')}")

    async def download_seall(self, sdate, edate, monthly=True):
        """Download delist data and distcd from m(d)seall.
        Delist can be obtained from either mseall or msedelist. I use mseall since it contains exchcd, which is used
        when replacing missing dlret.
        shrcd and exchcd in mseall are usually those before halting/suspension. If a stock in NYSE is halted, exchcd
        in msenames can be -2, whereas that in mseall is 1.

        Args:
            sdate: start date. e.g., '2000-01-01'
            edate: end date
            monthly: If True download mseall else dseall.
        """
        f = 'm' if monthly else 'd'
        elapsed_time(f'Downloading {f}seall...')

        sql = f"""
            SELECT distinct permno, date, 
            dlret, dlstcd, shrcd, exchcd, 
            distcd, divamt  
            FROM crsp.{f}seall
            WHERE date between '{sdate}' and '{edate}'
        """

        data = self.db.raw_sql(sql)

        if monthly:
            data = self.convert_types(data, [('crsp', f'{f}seall')])
        else:
            data = self.convert_types(data, [('crsp', f'{f}seall')], np.float32)
        data.date = pd.to_datetime(data.date)

        self.save_data(data, f"{f}seall_{sdate.replace('-', '')}_{edate.replace('-', '')}", 'tmp')
        log(f"Download complete: {data.date.min().strftime('%Y-%m-%d')}-{data.date.max().strftime('%Y-%m-%d')}")

    async def download_funda(self, sdate, edate):
        """Download comp.funda.

        Args:
            sdate: start date. e.g., '2000-01-01'
            edate: end date
        """
        log(f'Downloading funda: {sdate}-{edate}')
        await asyncio.sleep(.01)

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
            AND datadate between '{sdate}' and '{edate}'
            ORDER BY c.gvkey, datadate
            """

        data = self.db.raw_sql(sql)
        if data.empty:
            return

        data = self.convert_types(data, [('comp', 'funda'), ('comp', 'company')])
        data.datadate = pd.to_datetime(data.datadate)

        self.save_data(data, f"funda_{sdate.replace('-', '')}_{edate.replace('-', '')}", 'tmp')
        log(f"Download complete: {data.datadate.min().strftime('%Y-%m-%d')}-{data.datadate.max().strftime('%Y-%m-%d')}")

    async def download_fundq(self, sdate, edate):
        """Download comp.fundq.

        Args:
            sdate: start date. e.g., '2000-01-01'
            edate: end date
        """
        log(f'Downloading fundq: {sdate}-{edate}')
        await asyncio.sleep(.01)

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
            AND datadate between '{sdate}' and '{edate}'
            ORDER BY c.gvkey, datadate
            """

        data = self.db.raw_sql(sql)
        if data.empty:
            return

        data = self.convert_types(data, [('comp', 'fundq'), ('comp', 'company')])
        data.datadate = pd.to_datetime(data.datadate)

        self.save_data(data, f"fundq_{sdate.replace('-', '')}_{edate.replace('-', '')}", 'tmp')
        log(f"Download complete: {data.datadate.min().strftime('%Y-%m-%d')}-{data.datadate.max().strftime('%Y-%m-%d')}")

    async def download_secd(self, sdate, edate):
        """Download comp.fundq.

        Args:
            sdate: start date. e.g., '2000-01-01'
            edate: end date
        """
        log(f'Downloading secd: {sdate}-{edate}')
        await asyncio.sleep(.01)

        sql = f"""
            SELECT gvkey, datadate, prccd, ajexdi, cshoc, iid
            FROM 
                comp.secd
            WHERE datadate between '{sdate}' and '{edate}'
            ORDER BY gvkey, datadate
            """

        data = self.db.raw_sql(sql)
        if data.empty:
            return

        data = self.convert_types(data, [('comp', 'secd')])
        data.datadate = pd.to_datetime(data.datadate)

        self.save_data(data, f"secd_{sdate.replace('-', '')}_{edate.replace('-', '')}", 'tmp')
        log(f"Download complete: {data.datadate.min().strftime('%Y-%m-%d')}-{data.datadate.max().strftime('%Y-%m-%d')}")

    async def download_g_secd(self, sdate, edate):
        """Download comp.fundq.

        Args:
            sdate: start date. e.g., '2000-01-01'
            edate: end date
        """
        log(f'Downloading g_secd: {sdate}-{edate}')
        await asyncio.sleep(.01)

        sql = f"""
            SELECT gvkey, datadate, prccd, ajexdi, cshoc, iid
            FROM 
                comp.g_secd
            WHERE datadate between '{sdate}' and '{edate}'
            ORDER BY gvkey, datadate
            """

        data = self.db.raw_sql(sql)
        if data.empty:
            return

        data = self.convert_types(data, [('comp', 'g_secd')])
        data.datadate = pd.to_datetime(data.datadate)

        self.save_data(data, f"g_secd_{sdate.replace('-', '')}_{edate.replace('-', '')}", 'tmp')
        log(f"Download complete: {data.datadate.min().strftime('%Y-%m-%d')}-{data.datadate.max().strftime('%Y-%m-%d')}")

    def download_async(self, table, sdate=None, edate=None, interval=5):
        """Asynchronous download. Split the total period into 'interval' years and download data for each sub-period
        asynchronously. If download fails, it can be started from the failed date: already downloaded files will be
        gathered together.

        Support tables: crspm, crspd, funda, fundq

        Args:
            table: table to download, eg, 'crspm'
            sdate: start date ('yyyy-mm-dd'). If None, '1925-01-01'.
            edate: end date. If None, today.
            interval: sub-period size in years.
        """
        elapsed_time(f'Asynchronous download: {table}')

        tmp_dir = config.rawdata_dir + 'tmp'  # sub-period files are temporarily saved here.

        sdate = sdate or '1925-01-01'
        ldate = edate or str(dt.datetime.now())[:10]  # last date
        edate = str(int(sdate[:4]) + interval) + sdate[4:]

        dates = []
        while sdate < ldate:
            dates.append((sdate, edate))
            sdate = edate
            edate = min(str(int(sdate[:4]) + interval) + sdate[4:], ldate)

        if table == 'msf':
            fcn = getattr(self, 'download_sf')
            kwargs = {'monthly': True}
        elif table == 'dsf':
            fcn = getattr(self, 'download_sf')
            kwargs = {'monthly': False}
        elif table == 'mseall':
            fcn = getattr(self, 'download_seall')
            kwargs = {'monthly': True}
        elif table == 'dseall':
            fcn = getattr(self, 'download_seall')
            kwargs = {'monthly': False}
        else:
            fcn = getattr(self, f'download_{table}')
            kwargs = {}
            # raise ValueError(f'Async download is not supported for table: {table}.')

        loop = asyncio.get_event_loop()
        # aws = [loop.run_in_executor(None, self.download_crspd_, sdate, edate) for sdate, edate in dates]
        aws = [fcn(sdate, edate, **kwargs) for sdate, edate in dates]
        loop.run_until_complete(asyncio.gather(*aws))

        # Read saved data files and save them as one big file.
        log('Gathering files...')
        data = []
        for f in os.listdir(tmp_dir):
            fpath = os.path.join(tmp_dir, f)
            if os.path.isfile(fpath) and (f[:len(table)] == table):
                data.append(pd.read_pickle(fpath))
        data = pd.concat(data)

        elapsed_time(f'Download complete.')
        if table in ('msf', 'dsf', 'mseall', 'dseall'):
            self.save_data(data, table, 'crsp')
            inspect_data(data, date_col='date', id_col='permno')
        elif table in ('funda', 'fundq', 'secd', 'g_secd'):
            self.save_data(data, table, 'comp')
            inspect_data(data, date_col='datadate', id_col='gvkey')

        shutil.rmtree(tmp_dir, ignore_errors=True)  # delete temporary folder.

    def download_all(self):
        """Download all tables."""
        # wrds.db.create_pgpass_file()  # if pgpass file doesn't exist yet.

        self.download_async('funda')
        self.download_async('fundq')
        self.download_async('msf')
        self.download_async('dsf')
        self.download_async('mseall')
        self.download_async('dseall')

        self.download_table('crsp', 'ccmxpf_linktable')
        self.download_table('crsp', 'mcti')
        self.download_table('ff', 'factors_monthly')
        self.download_table('ff', 'factors_daily')
        self.download_table('comp', 'exrt_dly')

        # self.download_table('ibes', 'recdsum')
        # self.download_table('comp', 'adsprate')
        # self.download_table('wrdsapps', 'ibcrsphist')


    ################################################################################
    #
    # UTILITIES
    #
    ################################################################################
    def merge_crsp_with_seall(self, crsp, monthly=True, fill_method=1):
        """Adjust crspm(d) return (ret) with delistm(d) return (dlret). The adjusted data is saved in crspm(d).pickle.
        Also, add distcd to crspm(d).

        Args:
            monthly: if True, adjust crspm.ret, else crspd.ret.
            fill_method: method to fill missing dlret. 0: don't fill, 1 or 2: see the code.
            keep_rawdata: If True, the unadjusted crspm(d) is saved in crspm(d)_raw.pickle, else, it is replaced by
            the adjusted data.
        """
        f = 'm' if monthly else 'd'

        seall = self.read_data(f'{f}seall')
        elapsed_time(f'Merging crsp{f} with {f}seall...')

        #######################################
        # DLRET
        #######################################
        delist = seall[['permno', 'date', 'dlret', 'dlstcd', 'shrcd', 'exchcd']].dropna(subset=['dlstcd']).drop_duplicates().copy()
        # elapsed_time(f'Merging crsp{f} with delist{f}...')
        log(f'crsp{f} shape: {crsp.shape}, delist{f} shape: {delist.shape}')

        # Fill missing delist return
        log(f'Missing dlret: {delist.dlret.isna().sum()}')
        if fill_method == 1:  # from JKP SAS code
            ridx = delist.dlret.isnull() & ((delist.dlstcd == 500) | ((delist.dlstcd >= 520) & (delist.dlstcd <= 584)))
            delist.loc[ridx, 'dlret'] = -0.30
        elif fill_method == 2:  # from GHZ SAS code
            # http://www.crsp.com/products/documentation/delisting-codes
            ridx = delist.dlret.isnull() & ((delist.dlstcd == 500) | ((delist.dlstcd >= 520) & (delist.dlstcd <= 584)))
            delist.loc[ridx & ((delist.exchcd == 1) | (delist.exchcd == 2)), 'dlret'] = -0.35
            delist.loc[ridx & (delist.exchcd == 3), 'dlret'] = -0.55
        else:  # do nothing
            pass

        log(f'Missing dlret after filling: {delist.dlret.isna().sum()}')

        # Merge crsp with delist
        crsp = pd.merge(crsp, delist[['permno', 'date', 'dlret']], on=['permno', 'date'], how='left')

        is_ret_nan = crsp.ret.isna() & crsp.dlret.isna()
        crsp.ret = (1 + crsp.ret.fillna(0)) * (1 + crsp.dlret.fillna(0)) - 1
        crsp.loc[is_ret_nan, 'ret'] = np.nan  # Set ret = None if both ret and dlret missing.

        if not monthly:  # if crspd, stop here: no need to add cash dividend.
            elapsed_time(f'crsp{f} and {f}seall merged. crsp shape: {crsp.shape}')
            return crsp

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
        crsp = pd.merge(crsp, dist, on=['permno', 'date'], how='left')

        # self.save_data(crsp, f'crsp{f}')
        elapsed_time(f'crsp{f} and {f}seall merged. crsp shape: {crsp.shape}')
        return crsp

    @staticmethod
    def add_gvkey_to_crsp(crsp):
        """Add gvkey to crsp using ccmxpf_linktable and identify primary stocks.
        There are two tables we can use to link permno with gvkey: ccmxpf_linktable and ccmxpf_linkhist.
        ccmxpf_lnkused is simply a merge table of ccmxpf_lnkhist and ccmxpf_lnkused.
        https://wrds-www.wharton.upenn.edu/pages/support/research-wrds/macros/wrds-macros-cvccmlnksas/

        Ref:
            https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp-and-compustat/

        Args:
            crsp: crspm(d) Dataframe with index=date/permno.

        Returns:
            crspm(d) with gveky and primary (primary stock indicator) columns added. Unlinked rows are dropped.
        """
        elapsed_time('Adding gvkey to crsp...')

        link = WRDS.read_data('ccmxpf_linktable', 'crsp')
        link = link[~link.lpermno.isna()]
        link = link[link.usedflag == 1]
        link = link[link.linktype.isin(['LC', 'LU', 'LS'])]  # primary links only
        # link = link[link.linktype.isin(['LC', 'LN', 'LU', 'LX', 'LD', 'LS'])]  # primary and secondary links
        link = link[['lpermno', 'gvkey', 'linkdt', 'linkenddt', 'linkprim']].rename(columns={'lpermno': 'permno'})
        link.linkdt = pd.to_datetime(link.linkdt)
        link.linkenddt = pd.to_datetime(link.linkenddt)

        # Exceptions
        link.loc[(link.gvkey == '002759') & (link.linkdt == '1985-09-26') & (link.permno == 66843), 'linkprim'] = 'C'
        link.loc[(link.gvkey == '013699') & (link.linkdt == '1988-08-25') & (link.permno == 75189), 'linkprim'] = 'C'
        link.loc[(link.gvkey == '004162') & (link.linkdt == '2002-10-01') & (link.permno == 31318), 'linkprim'] = 'C'

        # To use merge_asof, data should be sorted.
        # crsp.reset_index(inplace=True)
        crsp.sort_values('date', inplace=True)
        link.sort_values('linkdt', inplace=True)

        # merge crsp with link. linkdt <= date <= linkenddt
        crsp['permno'] = crsp['permno'].astype('Int64')
        link['permno'] = link['permno'].astype('Int64')
        crsp = pd.merge_asof(crsp, link, left_on='date', right_on='linkdt', by='permno')
        crsp.loc[crsp.date > crsp.linkenddt, ['gvkey', 'linkprim']] = None

        # Set primary stock indicator.
        # 1. If there's only one permno matched to a gvkey, set it as primary.
        # 2. elif linkprim == 'P' or 'C', set as primary.
        # 3. If primary is still unidentified, set the security with max vol as primary (JKP)
        #
        # NOTE
        # 1. I use (1) since linkprim is not 100% reliable: eg, permno (15424) and gvkey (174314) are 1:1 mapping but
        #    linkprim changes from J to P on 2018-05-01.
        # 2. I group by gvkey rather than permco since there are cases where multiple gvkeys are mapped to one permco
        #    in a given month: eg, permco = 8243.

        #     primary = np.zeros(crsp.shape[0])
        #     primary[crsp.linkprim.isin(['P', 'C'])] = 1
        #     dvol = (crsp.prc.abs() * crsp.vol).values  # dollar volume
        #
        #     gsize = crsp.groupby(['date', 'permco']).size().values
        #
        #     @njit
        #     def fcn(primary, dvol, gsize):
        #         idx0 = 0
        #         for i in gsize:
        #             primary_ = primary[idx0:idx0 + i]
        #             if not np.any(primary_ == 1):  # no primary yet
        #                 if i == 1: # only one permno
        #                     primary_[:] = 1
        #                 else:
        #                     # primary_[dvol[idx0:idx0 + i] == np.max(dvol[idx0:idx0 + i])] = 1
        #                     primary_[np.argmax(dvol[idx0:idx0 + i])] = 1
        #             idx0 += i
        #
        #         return primary
        #
        #     crsp['primary'] = fcn(primary, dvol, gsize).astype(bool)

        crsp['primary'] = False
        crsp.loc[crsp.linkprim.isin(['P', 'C']), 'primary'] = True
        crsp.loc[crsp.groupby(['date', 'gvkey']).permno.transform('size') == 1, 'primary'] = True  # Primary if permno and gveky are 1:1

        # Primary identification by volume
        crsp['dvol'] = crsp.prc.abs() * crsp.vol  # dollar volume
        crsp['max_dvol'] = crsp.groupby(['date', 'gvkey'])['dvol'].transform(max)  # JKP groups by permco not gvkey
        crsp['nprimary'] = crsp.groupby(['date', 'gvkey']).primary.transform(sum)  # num. primary (0 means unidentified)
        crsp.loc[(crsp.nprimary == 0) & (crsp.dvol == crsp.max_dvol), 'primary'] = True

        crsp['max_dvol'] = crsp.groupby(['date', 'permco'])['dvol'].transform(max)  # JKP groups by permco not gvkey
        crsp['nprimary'] = crsp.groupby(['date', 'permco']).primary.transform(sum)  # num. primary (0 means unidentified)
        crsp.loc[crsp.gvkey.isna() & (crsp.nprimary == 0) & (crsp.dvol == crsp.max_dvol), 'primary'] = True

        # Check missing or dup
        nprimary = crsp.groupby(['date', 'gvkey']).primary.sum()
        if (nprimary != 1).any():
            log('Primary missing or dup.')
            log(nprimary[nprimary != 1])

        crsp.drop(columns=['linkdt', 'linkenddt', 'linkprim', 'dvol', 'max_dvol', 'nprimary'], inplace=True)
        crsp.sort_values(['permno', 'date'], inplace=True)
        # crsp.set_index(['date', 'permno'], inplace=True)

        elapsed_time(f'gvkey added to crsp: crsp shape: {crsp.shape}, missing gvkey: {crsp.gvkey.isna().sum()}')
        log(f'No. unique permnos: {len(crsp.permno.unique())}, no. unique gvkeys: {len(crsp.gvkey.unique())}')
        return crsp

    def preprocess_crsp(self):
        """Add delist return, gveky, and primary indicator to crsp."""

        crsp = self.read_data('msf')
        crsp = self.merge_crsp_with_seall(crsp, monthly=True)
        crsp = self.add_gvkey_to_crsp(crsp)
        self.save_data(crsp, 'crspm')

        crsp = self.read_data('dsf')
        crsp = self.merge_crsp_with_seall(crsp, monthly=False)
        crsp = self.add_gvkey_to_crsp(crsp)
        self.save_data(crsp, 'crspd')

    @staticmethod
    def save_data(data, table_name, library=None):
        """Save data in pickle format. A pickle file preserves data types and is much faster to read compared to a csv
        file. data is saved in the following location:
            rawdata_dir/table_name.pickle if library=None,
            rawdata_dir/library/table_name.pickle otherwise.

        Args:
            data: data to save (Dataframe)
            table_name: file name
            library: folder
        """
        fdir = config.rawdata_dir + (library + '/' if library else '')

        if not os.path.isdir(fdir):
            os.makedirs(fdir)

        data.to_pickle(fdir + table_name + '.pickle')

    @staticmethod
    def read_data(table_name, library=None, index_col=None, sort_col=None):
        """Read data from library/table_name.pickle. If library is None, all folders under rawdata_dir will be searched.

        Args:
            table_name: file name
            library: folder
            index_col: (list of) column(s) to be set as index.
            sort_col: (list of) column(s) to sort data

        Returns:
            data read.
        """
        data = None

        if library:
            fpath = config.rawdata_dir + library + '/' + table_name + '.pickle'
            data = pd.read_pickle(fpath)
        else:  # if library is None, search all subdirectories.
            for dir in os.walk(config.rawdata_dir):
                fpath = dir[0] + '/' + table_name + '.pickle'
                if os.path.exists(fpath):
                    data = pd.read_pickle(fpath)
        if data is None:
            raise ValueError(f'[{table_name}] does not exist.')
        if sort_col is not None:
            data.sort_values(sort_col, inplace=True)
        if index_col is not None:
            data.set_index(index_col, inplace=True)

        return data

    @staticmethod
    def save_as_csv(table_name, library=None, fpath=None):
        """Read table_name and save it as a csv file.

        Args:
            table_name: file name
            library: folder
            fpath: file path to save the csv file. If None, the file is saved as library/table_name.csv
        """
        fpath = fpath or config.rawdata_dir + (library + '/' if library else '') + table_name + '.csv'
        data = WRDS.read_data(table_name, library)
        data.to_csv(fpath)

    @staticmethod
    def get_risk_free_rate(sdate=None, edate=None, src='mcti'):
        """Get risk free rate.

        Args:
            sdate: start date
            edate: end date
            src: data source: 'mcti' or 'ff_monthly'.

        Returns:
            risk-free rate Dataframe with index=date, columns=rf.
        """
        if src == 'mcti':
            rf = WRDS.read_data('mcti')[['caldt', 't30ret']]
            rf['caldt'] = pd.to_datetime(rf.caldt)
            rf = rf.rename(columns={'caldt': 'date', 't30ret': 'rf'}).set_index('date')
        elif src == 'ff':
            rf = WRDS.read_data('ff_monthly')[['dateff', 'rf']]
            rf['dateff'] = pd.to_datetime(rf.dateff)
            rf = rf.rename(columns={'dateff': 'date'}).set_index('date')
        else:
            raise ValueError(f'Undefined risk free data source: {src}')

        return rf.loc[sdate:edate]

    @staticmethod
    def convert_fund_currency_to_usd(fund, table='funda'):
        curcol = 'curcd' if table == 'funda' else 'curcdq'

        exrt = WRDS.read_data('exrt_dly')
        exrt.datadate = pd.to_datetime(exrt.datadate)
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

    ################################################################################
    #
    # UNDER CONSTRUCTION
    #
    ################################################################################
    def download_ibes(self):
        log('This function is under construction.')
        return

        log('downloading ibes...')
        sql = """
            SELECT 
               ticker, cusip, fpedats, statpers, anndats_act, numest, anntims_act, medest, actual, stdev
            FROM 
                ibes.statsum_epsus
            /*WHERE fpi='6'   1 is for annual forecasts, 6 is for quarterly */
            WHERE statpers < anndats_act  /* only keep summarized forecasts prior to earnings annoucement */
            AND measure = 'EPS' 
            AND medest is not null 
            AND fpedats is not null
            AND fpedats >= statpers
            ORDER BY cusip, fpedats, statpers
            """

        data = self.db.raw_sql(sql)
        data.fpedats = pd.to_datetime(data.fpedats)
        save_data(data, 'ibes')
        log(f'download complete.')

    def add_permno_to_fund(self, fd, dropna=True):
        log('This function is under construction.')
        return

        if 'permno' in fd.columns:
            return fd

        link = read_data('ccmxpf_linktable', 'crsp')
        link = link[~link.lpermno.isna()]
        link = link[link.usedflag == 1]
        link = link[link.linktype.isin(['LC', 'LN', 'LU', 'LX', 'LD', 'LS'])]
        link = link[['lpermno', 'gvkey', 'linkdt', 'linkenddt']].rename(columns={'lpermno': 'permno'})

        link.linkdt = pd.to_datetime(link.linkdt)
        link.linkenddt = pd.to_datetime(link.linkenddt)

        fd = fd.sort_index().reset_index()
        link = link.sort_values(['linkdt', 'gvkey'])

        fd = pd.merge_asof(fd, link, left_on='datadate', right_on='linkdt', by='gvkey')
        # fd = fd[(fd.datadate <= fd.linkenddt) | fd.linkenddt.isna()]
        fd.loc[fd.datadate > fd.linkenddt, 'permno'] = None

        fd.drop(columns=['linkdt', 'linkenddt'], inplace=True)

        fd = fd.sort_values(['gvkey', 'datadate']).set_index(['datadate', 'gvkey'])
        if dropna:
            log(f'fd size: {len(fd)}')
            fd = fd[~fd.permno.isna()]
            log(f'fd size with not missing permno: {len(fd)}')
        return fd

    def merge_ibes_crspm(self, ibes, crspm):
        """
        https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-ibes-and-crsp-data/
        * Merging IBES and CRSP datasets using ICLINK table;
        proc sql;
        create table IBES_CRSP
        as select a.ticker, a.STATPERS, a.meanrec, c.permno, c.date, c.ret from ibes.recdsum as a,
        home.ICLINK as b,
        crsp.msf as c
        where a.ticker=b.ticker and b.permno=c.permno and intnx('month',a.STATPERS,0,'E') = intnx('month',c.date,0,'E');
        quit;
        """
        log('This function is under construction.')
        return

        db = sqlite3.connect(':memory:')

        link = read_data('ibcrsphist', 'wrdsapp')

        # write the tables
        ibes.reset_index()[['cusip', 'fpedats']].to_sql('ibes', db, index=False)
        link.to_sql('link', db, index=False)
        crspm.reset_index()[['date', 'permno']].to_sql('crspm', db, index=False)
        elapsed_time('tables uploaded to db')

        # Merge ibes with link
        sql = """
            CREATE TABLE temp as
            SELECT l.lpermno as permno, f.*
            FROM
                ibes as f, link as l
            /*ON f.cusip = l.lcusip
            AND f.fpedats >= sdate 
            AND f.fpedats <= edate
            WHERE l.lpermno is not null*/
            """

        db.execute(sql)
        elapsed_time('funda and link merged')

        sql = """
            SELECT c.*, f.*
            FROM
                crspm as c 
            LEFT JOIN
                temp as f
            ON c.permno = f.permno
            AND c.date >= f.datadate + 7
            AND c.date < f.datadate + 20
            """

        table = pd.read_sql_query(sql, db)
        return table


if __name__ == '__main__':
    os.chdir('../')

    wrds = WRDS('fehouse')
    # # wrds.download_table('comp', 'g_secm')
    # wrds.download_async('dsf')
    # # wrds.download_all()
    # wrds.preprocess_crsp()

    wrds.download_async('msf')

