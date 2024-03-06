"""Firm characteristics generation

This example shows the full process of i) data download, ii) factor generation, and iii) firm characteristic
generation.
In this example,

    - Funda is merged with fundq to create quarterly updated annual accounting data.
    - Funda and fundq are assumed to be available 4 months later.
    - The latest market equity (me) is used when generating firm characteristics.
    - All firm characteristics defined in PyAnomaly are generated.

Generated factors are saved to './output/factors_monthly' and './output/factors_daily', and firm characteristics are
saved to './output/merge'.

Processing time: approx. 45 minutes to download data and 15 minutes for the rest.
"""

from pyanomaly.globals import *
from pyanomaly.wrdsdata import WRDS
from pyanomaly.characteristics import FUNDA, FUNDQ, CRSPM, CRSPD, Merge
from pyanomaly.factors import make_all_factor_portfolios
from pyanomaly.analytics import *


def example1():

    # Set log file path as config.log_dir + this module name.
    set_log_path(__file__)  # You can also give a different log file name: e.g., set_log_path('logfile.log')

    # Start timer.
    start_timer('Example 1')

    # To replicate JKP's SAS code as closely as possible, uncomment the following line.
    # set_config(replicate_jkp=True)

    ###################################
    # Download data frm WRDS.
    ###################################
    drawline()  # This will print a line in the log file. Only for visual effect.
    log('DOWNLOADING DATA')
    drawline()

    wrds = WRDS(wrds_username)  # Use your WRDS user id.

    # Download all necessary data asynchronously.
    # If network is slow or memory is insufficient, set run_in_executer=False.
    wrds.download_all(run_in_executer=True)

    # Create crspm(d) from m(d)sf and m(d)seall and add gvkey to them.
    wrds.preprocess_crsp()

    ###################################
    # Generate factors
    ###################################
    drawline()
    log('FACTOR PORTFOLIOS')
    drawline()

    # Make factor portfolios: FF3, FF5, HXZ4, and SY4.
    # These are used in some firm characteristics, such as the residual momentum.
    make_all_factor_portfolios(monthly=True, daily=True)  # Monthly and daily factors

    ###################################
    # Generate firm characteristics
    ###################################
    # To generate characteristics in a certain column of mapping.xlsx, set ``alias`` = column name.
    # For instance, ``alias = 'jkp'`` will generated firm characteristics defined in 'jkp' column.
    # Setting ``alias = None`` will generate all available firm characteristics.
    # Note that some firm characteristics will be created twice using different implementations.
    alias = None

    # Start date ('yyyy-mm-dd'). Set to None to create characteristics from as early as possible.
    sdate = None

    ###################################
    # CRSPM
    ###################################
    # Generate firm characteristics from crspm.
    drawline()
    log('PROCESSING CRSPM')
    drawline()

    crspm = CRSPM(alias=alias)

    # Load crspm.
    crspm.load_data(sdate)

    # Filter data: shrcd in [10, 11, 12]
    crspm.filter_data()

    # Fill missing months by populating the data.
    crspm.populate(method=None)

    # Some preprocessing, e.g., creating frequently used variables.
    crspm.update_variables()

    # Merge crspm with the monthly FF3 factors generated earlier.
    crspm.merge_with_factors(['mktrf', 'smb_ff', 'hml'])

    # Display the firm characteristics to be generated.
    crspm.show_available_chars()

    # Create characteristics.
    crspm.create_chars()

    # Postprocessing: delete temporary variables and inspect the generated data.
    crspm.postprocess()

    # Save crspm to use in other examples.
    crspm.save()

    ###################################
    # CRSPD
    ###################################
    # Generate firm characteristics from crspd.
    drawline()
    log('PROCESSING CRSPD')
    drawline()

    crspd = CRSPD(alias=alias)

    # Load crspd.
    crspd.load_data(sdate)

    # Filter data: shrcd in [10, 11, 12]
    crspd.filter_data()

    # Some preprocessing, e.g., creating frequently used variables.
    crspd.update_variables()

    # Merge crspd with the daily FF3 and HXZ4 factors generated earlier.
    crspd.merge_with_factors(['mktrf', 'smb_ff', 'hml', 'smb_hxz', 'inv', 'roe'])

    # Display the firm characteristics to be generated.
    crspd.show_available_chars()

    # Create characteristics.
    crspd.create_chars()

    # Postprocessing: delete temporary variables and inspect the generated data.
    crspd.postprocess()

    # Delete raw crspd data to save memory.
    del crspd.cd

    ###################################
    # FUNDQ
    ###################################
    # Generate firm characteristics from fundq.
    drawline()
    log('PROCESSING FUNDQ')
    drawline()

    fundq = FUNDQ(alias=alias)

    # Load fundq.
    fundq.load_data(sdate)

    # fundq has some duplicates (same datedate/gvkey). Drop duplicates.
    fundq.remove_duplicates()

    # Convert values in another currency (currently only CAD) to USD.
    fundq.convert_currency()

    # Populate data to monthly.
    # limit=3: Forward fill up to 3 months.
    # lag=4: Shift data 4 months, i.e., data is available 4 months later.
    # new_date_col='date': Populated data has index 'date'/'gvkey', and 'datadate' becomes a column.
    fundq.populate(MONTHLY, limit=3, lag=4, new_date_col='date')

    # Make quarterly variables from ytd variables and use them to fill missing quarterly variables.
    fundq.create_qitems_from_yitems()

    # Some preprocessing, e.g., creating frequently used variables.
    fundq.update_variables()

    # Display the firm characteristics to be generated.
    fundq.show_available_chars()

    # Create characteristics.
    fundq.create_chars()

    # Postprocessing: delete temporary variables and inspect the generated data.
    fundq.postprocess()

    ###################################
    # FUNDA
    ###################################
    # Generate firm characteristics from funda.
    drawline()
    log('PROCESSING FUNDA')
    drawline()

    funda = FUNDA(alias=alias)

    # Load funda.
    funda.load_data(sdate)

    # Convert values in another currency (currently only CAD) to USD.
    funda.convert_currency()

    # Populate data to monthly.
    # limit=12: Forward fill up to 12 months.
    # lag=4: Shift data 4 months, i.e., data is available 4 months later.
    # new_date_col='date': Populated data has index 'date'/'gvkey', and 'datadate' becomes a column.
    funda.populate(MONTHLY, limit=12, lag=4, new_date_col='date')

    # Generate quarterly-updated funda data from fundq and merge them with funda.
    funda.merge_with_fundq(fundq)

    # Some preprocessing, e.g., creating frequently used variables.
    funda.update_variables()

    # Add the market equity of crspm to funda.
    funda.add_crsp_me(crspm)

    # Display the firm characteristics to be generated.
    funda.show_available_chars()

    # Create characteristics.
    funda.create_chars()

    # Postprocessing: delete temporary variables and inspect the generated data.
    funda.postprocess()

    ###################################
    # Merge
    ###################################
    # Merge FUNDA, FUNDQ, CRSPM, and CRSPD and generate firm characteristics that require data from multiple data
    # sources.
    drawline()
    log('PROCESSING MERGE')
    drawline()

    merge = Merge(alias)

    # Merge all data together.
    merge.preprocess(crspm, crspd, funda, fundq)

    # Display the firm characteristics to be generated.
    merge.show_available_chars()

    # Create characteristics.
    merge.create_chars()

    # Postprocessing: delete temporary variables and inspect the generated data.
    merge.postprocess()

    # Save the results to config.output_dir + 'merge'.
    # The default behavior of Panel.save() is to save all the columns of Panel.data.
    # You can reduce file size by choosing columns to save. The following code saves firm characteristics + columns.
    columns = ['gvkey', 'datadate', 'primary', 'exchcd', 'me', 'ret', 'exret', 'rf']
    merge.save(other_columns=columns)

    elapsed_time('End of Example 1.')


if __name__ == '__main__':
    os.chdir('../')
    wrds_username = ''  # Your wrds username

    example1()
