"""Define a new characteristic

This example demonstrates how to define a new characteristic by inheriting CRSPM.

A new characteristic, 'excess_ret_change', will be defined, which is the excess return over the market return
divided by the one-year average excess return (I have no idea if this has any predictive power).

- It is assumed that data has been downloaded from WRDS.

Processing time: approx. 18 seconds.
"""

from pyanomaly.globals import *
from pyanomaly.characteristics import CRSPM
from pyanomaly.analytics import *


# Define a class by inheriting CRSPM.
class MyCRSPM(CRSPM):

    # Method for the new characteristic.
    # Method name should be of the form 'c_[characteristic name]'.
    def c_excess_ret_change(self):
        """Change of excess return over the market return."""
        # Make a short (one-line) docstring for description.
        # This is displayed when 'show_available_chars()' is called.

        cm = self.data

        # One-year average excess return
        avg_exret = self.rolling(cm.exret - cm.mktrf, 12, 'mean')

        # Excess return.
        exret = cm.exret - cm.mktrf

        # Characteristic
        char = exret / avg_exret

        return char

def example4():
    set_log_path(__file__)
    start_timer('Example 4')

    # Create a MyCRSPM instance.
    crspm = MyCRSPM()

    # Load crspm.
    crspm.load_data()

    # Filter data: shrcd in [10, 11, 12]
    crspm.filter_data()

    # Fill missing months by populating the data.
    crspm.populate(method=None)

    # Some preprocessing, e.g., creating frequently used variables.
    crspm.update_variables()

    # Merge crspm with the monthly factors generated earlier.
    # Need this as excess_ret_change uses the market return.
    crspm.merge_with_factors()

    # Display the firm characteristics to be generated.
    crspm.show_available_chars()

    # Generate excess_ret_change.
    # Without any argument, all characteristics defined in CRSPM and MyCRSPM will be generated.
    crspm.create_chars('excess_ret_change')

    # Some other characteristics defined in CRSPM can also be generated.
    crspm.create_chars(['ret_12_1', 'ret_1_0'])  # 12M momentum and short-term reversal.

    # Check the result
    print(crspm[['excess_ret_change', 'ret_12_1', 'ret_1_0']])

    # Postprocessing: delete temporary variables and inspect the generated data.
    crspm.postprocess()

    # Save the data (raw data + firm characteristics) to config.output_dir + 'mycrspm'.
    # To save only firm characteristics and a few columns of the raw data, use 'other_columns' argument.
    crspm.save()

    # The saved file can be loaded using crspm.load().
    crspm = MyCRSPM().load()
    print(crspm[['excess_ret_change', 'ret_12_1', 'ret_1_0']])

    elapsed_time('End of Example 4.')


if __name__ == '__main__':
    os.chdir('../')

    example4()
