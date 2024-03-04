"""This module defines functions to set/get package configuration.

    A configuration can be accessed by either ``get_config(attr)`` or ``config.attr``.

    **Configurations**

    .. csv-table::
        :header: Attribute, Description, Default value

        input_dir, Input file top-level directory, './input/'
        output_dir, Output file top-level directory, './output/'
        log_dir, Log file directory, './log/'
        mapping_file_path, Function-characteristic mapping file path, './mapping.xlsx'
        factors_monthly_fname, Monthly factor file name, 'factors_monthly'
        factors_daily_fname, Daily  factor file name, 'factors_daily'
        factor_names, "Factor name mapping dictionary. The keys are the factor names used in PyAnomaly and the values
        are the factor names in monthly(daily) factor file. This configuration can be used when factor files are
        obtained externally and have different factor names.", "{'rf': 'rf', 'mktrf': 'mktrf', 'smb_ff': 'smb_ff',
        'hml': 'hml', 'smb_ff5': 'smb_ff5', 'rmw': 'rmw', 'cma': 'cma', 'smb_hxz': 'smb_hxz', 'inv': 'inv',
        'roe': 'roe', 'smb_sy': 'smb_sy', 'mgmg': 'mgmt', 'perf': 'perf'}"
        replicate_jkp, Whether to replicate JKP version. True or False, False
        float_type, Float data type. 'float32' or 'float64', 'float64'
        file_format, File format used to save data. 'pickle' or 'parquet', 'pickle'
        disable_jit, Disable jitting. Applicable only to non-cached jitted functions., False
        jit_parallel, Enable parallel looping in jitted functions., True
        numba_num_threads, "Number of threads used in Numba parallel mode. Should be fewer than the number of CPU
        cores.", Number of CPU cores
        debug, Print debugging messages., False

    .. csv-table::
        :header: Factor model, Factors

        Fama and French 3-factor model, "mktrf, smb_ff, hml"
        Fama and French 5-factor model, "mktrf, smb_ff5, hml, rmw, cma"
        "Hou, Xu, and Zhang 4-factor model", "mktrf, smb_hxz, inv, roe"
        Stambaugh and Yuan 4-factor model, "mktrf, smb_sy, mgmt, perf"

    **Methods**

    .. autosummary::
        :nosignatures:

        set_config
        get_config
"""

import numpy as np
import pandas as pd
import numba as nb
from pyanomaly.datatypes import struct

########################################
# General configuration
########################################
config = struct(
    # File directories and names.
    input_dir='./input/',
    output_dir='./output/',
    log_dir = './log/',
    mapping_file_path='./mapping.xlsx',

    factors_monthly_fname='factors_monthly',
    factors_daily_fname='factors_daily',

    # keys: factor names in pyanomaly
    # values: factor names in monthly(daily)_factor file.
    factor_names={
        'rf': 'rf',
        'mktrf': 'mktrf',
        'smb_ff': 'smb_ff',
        'hml': 'hml',
        'smb_ff5': 'smb_ff5',
        'rmw': 'rmw',
        'cma': 'cma',
        'smb_hxz': 'smb_hxz',
        'inv': 'inv',
        'roe': 'roe',
        'smb_sy': 'smb_sy',
        'mgmg': 'mgmt',
        'perf': 'perf',
    },

    # True to replicate JKP version (not completely). Verification purpose only.
    replicate_jkp=False,

    # Float data type.
    float_type='float64',

    # file type
    file_format='pickle',

    # Numba
    disable_jit=False,
    jit_parallel=True,
    numba_num_threads=nb.config.NUMBA_NUM_THREADS,

    # Print debugging messages.
    debug=False,
)


def set_config(**kwargs):
    """Set configuration.

    Args:
        **kwargs: Keword arguments of configuration attributes and their values.

    Examples:
        Change the float type to 'float32'.

        >>> set_config(float_type='float32')

        Set input and output directories to './my_input/' and './my_output/', respectively.

        >>> set_config(input_dir='./my_input/', out_dir='./my_output')
    """

    for k, v in kwargs.items():
        if (k[-4:] == '_dir') and (v[-1] != '/'):
            v += '/'

        # Check if the value is valid.
        if k == 'fioat_type':
            assert v in ('float32', 'float64')
        elif k == 'file_format':
            assert v in ('parquet', 'pickle')
        elif k in ('replicate_jkp', 'jit_parallel', 'disable_jit', 'debug'):
            assert isinstance(v, bool)
        elif k == 'numba_num_threads':
            assert v <= nb.config.NUMBA_NUM_THREADS
            nb.set_num_threads(v)

        config[k] = v


def get_config(attr):
    """Get configuration.

    Args:
        attr: String. A configuration attribute.

    Returns:
        The value of the attribute.

    Examples:
        >>> get_config('input_dir')
        './input/'
    """

    return config[attr]


########################################
# Display/Stdout
########################################
# pandas
pd.set_option('display.min_rows', 100, 'display.max_rows', 100, 'display.max_columns', 1000,
              'display.float_format', '{:.3f}'.format, 'display.width', 320, 'display.max_colwidth', 100)

# pd.set_option('mode.copy_on_write', True)


########################################
# Warning suppression
########################################

from warnings import simplefilter, filterwarnings
simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# numpy
np.seterr(divide='ignore', invalid='ignore')  # Silence divide by 0 or nan warning.

# 'Mean of empty slice' warning occurs when trying to take a mean of all nans using np.nanmean. Ignore it.
filterwarnings(action='ignore', message='Mean of empty slice')

# numba
from numba.core.errors import NumbaTypeSafetyWarning

simplefilter('ignore', category=NumbaTypeSafetyWarning)


# matplotlib
# try:  # try to use latex font when possible.
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif', size=10)
# except:
#     pass

