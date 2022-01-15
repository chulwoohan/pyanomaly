import os
import numpy as np
import pandas as pd
from .datatypes import struct

########################################
# File IO
########################################
config = struct(
    input_dir='./input/',
    output_dir='./output/',
    mapping_file_path='./mapping.xlsx',

    factors_monthly_fname='factors_monthly',
    factors_daily_fname='factors_daily',
    # factor_name in pyanomaly: factor_name in monthly(daily)_factors file.
    factor_names={
        'rf': 'rf',
        'mktrf': 'mktrf',
        'smb_ff': 'smb_ff',
        'hml': 'hml',
        'inv': 'inv',
        'roe': 'roe',
        'smb_hxz': 'smb_hxz',
    },
    # True to replicate JKP version (not completely). Verification purpose only. Will be deprecated.
    REPLICATE_JKP=False

)

########################################
# Display/Stdout
########################################
# pandas
pd.set_option('display.min_rows', 100, 'display.max_rows', 100, 'display.max_columns', 1000,
              'display.float_format', '{:.3f}'.format, 'display.width', 320, 'display.max_colwidth', 100)

# numpy
np.seterr(divide='ignore', invalid='ignore')  # Silence divide by 0) or nan warning.

# matplotlib
# try:  # try to use latex font when possible.
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif', size=10)
# except:
#     pass


########################################
# Other configurations
########################################
