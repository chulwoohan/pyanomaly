import pyanomaly as pa
from pyanomaly.characteristics import *
from pyanomaly.jkp import make_factor_portfolios

"""
EXAMPLE 1

This example shows the full process of i) data download, ii) factor creation, and iii) characteristics creation.
In this example, 
    i) funda is merged with fundq before creating annual characteristics;
    ii) market equity (me) of the creation month from crspm is used.

"""

class myFUNDA1(FUNDA):
    """Derived class of FUNDA. You can call the methods in preprocess() below in main() instead of
    making this class. I do this to show you how to derive a class.
    """

    def preprocess(self, fundq, crspm):
        self.convert_currency()  # convert currency.
        self.convert_to_monthly()  # populate to monthly.
        self.merge_with_fundq(fundq)  # use fundq data if available.
        self.update_variables()  # preprocess variables.
        self.add_crsp_me(crspm)  # add crsp market equity.

def example1():
    elapsed_time()

    # # This needs to be done only once.
    # drawline()
    # log('DOWNLOADING DATA')
    # drawline()
    # wrds = WRDS('fehouse')
    # wrds.download_all()
    # wrds.preprocess_crsp()
    #
    # # This needs to be done only once.
    # drawline()
    # log('FACTOR PORTFOLIOS')
    # drawline()
    # make_factor_portfolios(monthly=True)
    # make_factor_portfolios(monthly=False)

    acronym = 'acronym_jkp'  # Generate all possible characteristics
    sdate = '1950-01-01'  # Set to None to create characteristics from as early as possible.

    drawline()
    log('PROCESSING CRSPM')
    drawline()
    crspm = CRSPM(acronym=acronym)
    crspm.load_raw_data(sdate)
    crspm.preprocess()
    crspm.show_available_functions()
    crspm.create_chars()
    crspm.postprocess()
    crspm.save_data()

    drawline()
    log('PROCESSING FUNDQ')
    drawline()
    fundq = FUNDQ(acronym=acronym)
    fundq.load_raw_data(sdate)
    fundq.preprocess()
    fundq.show_available_functions()
    fundq.create_chars()
    fundq.postprocess()
    fundq.save_data()

    drawline()
    log('PROCESSING FUNDA')
    drawline()
    funda = myFUNDA1(acronym=acronym)
    funda.load_raw_data(sdate)
    funda.preprocess(fundq, crspm)
    funda.show_available_functions()
    funda.create_chars()
    funda.postprocess()
    funda.save_data()

    drawline()
    log('PROCESSING CRSPD')
    drawline()
    crspd = CRSPD(acronym=acronym)
    crspd.load_raw_data(sdate)
    crspd.preprocess()
    crspd.show_available_functions()
    crspd_chars = crspd.get_available_chars()
    crspd.create_chars(crspd_chars)
    crspd.postprocess()
    crspd.save_data()

    # crspm = CRSPM(acronym=acronym)
    # crspm.load_data()
    #
    # crspd = CRSPD(acronym=acronym)
    # crspd.load_data()
    #
    # fundq = FUNDQ(acronym=acronym)
    # fundq.load_data()
    #
    # funda = myFUNDA1(acronym=acronym)
    # funda.load_data()

    drawline()
    log('PROCESSING MERGE')
    drawline()
    merge = Merged()
    merge.preprocess(crspm, crspd, funda, fundq)
    merge.show_available_functions()
    merge_chars = merge.get_available_chars()
    merge.create_chars(merge_chars)
    merge.postprocess()

    # if you want to keep only the characteristics
    # columns = ['gvkey', 'datadate_a', 'datadate_q'] + crspm_chars + crspd_chars + funda_chars + fundq_chars
    # merge.data = merge.data[columns]

    merge.save_data('merged2')

if __name__ == '__main__':
    # example1()

    acronym = 'acronym_jkp'  # Generate all possible characteristics

    crspm = CRSPM(acronym=acronym)
    crspm.load_data()

    crspd = CRSPD(acronym=acronym)
    crspd.load_data()

    fundq = FUNDQ(acronym=acronym, freq=MONTHLY)
    fundq.load_data()

    funda = myFUNDA1(acronym=acronym, freq=MONTHLY)
    funda.load_data()

    drawline()
    log('PROCESSING MERGE')
    drawline()
    merge = Merged()
    merge.preprocess(crspm, crspd, funda, fundq, delete_data=False)
    merge.show_available_functions()
    merge_chars = merge.get_available_chars()
    merge.create_chars(merge_chars)
    merge.postprocess()

    merge.save_data('merged3')
