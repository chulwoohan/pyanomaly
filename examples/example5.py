"""Firm characteristics using year-end ME

This example demonstrates how to generate funda firm characteristics using year-end ME instead of the latest ME.

    i) All firm characteristics defined in FUNDA are generated.
    ii) Only USD-denominated stocks are sampled: currency conversion is unnecessary.
    iii) Only funda is used without merging with fundq: merging the two datasets as in Example 1 is also possible.
    iv) It is assumed that funda data is available 6 months later.

- It is assumed that data has been downloaded from WRDS.

Processing time: approx. 105 seconds.
"""

from pyanomaly.characteristics import FUNDA, CRSPM
from pyanomaly.analytics import *


def example5():
    set_log_path(__file__)
    start_timer('Example 5')

    ###################################
    # Market equity (from CRSP)
    ###################################

    # Load crspm.
    crspm = CRSPM()
    crspm.load_data()

    # Filter data: shrcd in [10, 11, 12]
    crspm.filter_data()

    # Fill missing months by populating the data.
    crspm.populate(method=None)

    # Some preprocessing, e.g., creating frequently used variables.
    # 'me' and 'me_company' (firm-level me) are generated here.
    crspm.update_variables()

    ###################################
    # FUNDA firm characteristics
    ###################################

    # Load funda.
    funda = FUNDA()
    funda.load_data()

    # USD stocks only.
    funda.filter(('curcd', '==', 'USD'))

    # Populate data to monthly.
    # limit=12: Forward fill up to 12 months.
    # lag=6: Shift data 6 months, i.e., data is available 6 months later.
    funda.populate(MONTHLY, limit=12, lag=6, new_date_col='date')

    # Some preprocessing, e.g., creating frequently used variables.
    funda.update_variables()

    # Add year-end market equity to funda.
    funda.add_crsp_me(crspm, method='year_end')

    # Generate firm characteristics.
    funda.show_available_chars()
    funda.create_chars()

    # Postprocess and save results to './output/funda_yearend_me'.
    funda.postprocess()
    funda.save('funda_yearend_me')

    elapsed_time('End of Example 5.')


if __name__ == '__main__':
    os.chdir('../')

    example5()
