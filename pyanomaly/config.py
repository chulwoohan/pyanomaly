import os
import numpy as np
import pandas as pd
from .datatypes import struct

########################################
# File IO
########################################
config = struct(
    input_dir='./input/',
    rawdata_dir='./input/rawdata/',
    output_dir='./output/',
    daily_factors_fname='daily_factors_jkp',
    monthly_factors_fname='monthly_factors_jkp',
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
REPLICATE_JKP = True  # True to replicate JKP version (not completely)
