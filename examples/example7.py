"""Table download

This example demonstrates how to download a new table from WRDS.
Different ways of downloading comp.secm table are demonstrated.

The table will be saved to './input/comp/secm'.
"""

from pyanomaly.globals import *
from pyanomaly.wrdsdata import WRDS


def example7():
    set_log_path(__file__)
    start_timer('Example 4')

    wrds = WRDS(wrds_username)

    # Download the entire table at once.
    wrds.download_table('comp', 'secm', date_cols=['datadate'])  # 'datadate's type will be converted to datetime.

    # Download the entire table asynchronously.
    # Download data every `interval` years. For a small size data, this can be slower than `download_table()`.
    wrds.download_table_async('comp', 'secm', date_col='datadate', interval=5)

    # Download only some fields.
    sql = 'datadate, gvkey, cshoq, prccm'  # fields to select
    wrds.download_table_async('comp', 'secm', sql=sql, date_col='datadate')

    # Download data using a complete query statement.
    # Below is equivalent to the above.
    # Note that the query statement must contain 'WHERE [`date_col`] BETWEEN {} and {}'.
    sql = f"""
        SELECT datadate, gvkey, cshoq, prccm
        FROM comp.secm
        WHERE datadate between '{{}}' and '{{}}'
    """
    wrds.download_table_async('comp', 'secm', sql=sql)

    elapsed_time('End of Example 7.')


if __name__ == '__main__':
    os.chdir('../')
    wrds_username = ''  # Use your WRDS username

    example7()
