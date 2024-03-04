"""This module defines functions for file IO.

    .. autosummary::
        :nosignatures:

        write_to_file
        read_from_file
"""

from .globals import *

import pickle

def write_to_file(data, fname, fdir=None, typecast=True):
    """Write data to a file.

    The `data` is saved to `fdir/fname`. The file format is determined by ``config.file_format``.

    Args:
        data: DataFrame to save.
        fname: File name without extension.
        fdir: Directory. If None, ``config.output_dir``.
        typecast: If True, cast float to ``config.float_type`` and object to string before writing to a file.
    """
    fdir = fdir or config.output_dir

    if not os.path.isdir(fdir):
        os.mkdir(fdir)

    if typecast:
        for col in data:
            if is_float(data[col]) and (data[col].dtype != config.float_type):
                data[col] = data[col].astype(config.float_type)
            elif (data[col].dtype == 'object'):
                if is_bool_array(data[col]):
                    data[col] = data[col].fillna(False).astype(bool)
                else:
                    data[col] = data[col].astype('string', errors='ignore')

    if config.file_format == 'parquet':
        data.to_parquet(fdir + ('' if fdir[-1] == '/' else '/') + fname + '.parquet')
    else:
        data.to_pickle(fdir + ('' if fdir[-1] == '/' else '/') + fname + '.pickle')


def read_from_file(fname, fdir=None, typecast=True):
    """Read data from a file.

    Args:
        fname: File name without extension.
        fdir: Directory. If None, ``config.output_dir``.
        typecast: If True, cast float to ``config.float_type`` and object to string after reading from a file.

    Returns:
        DataFrame read from `fdir/fname`.
    """
    fdir = fdir or config.output_dir
    fpath = fdir + ('' if fdir[-1] == '/' else '/') + fname + '.' + config.file_format
    if config.file_format == 'parquet':
        data = pd.read_parquet(fpath)
    else:
        data = pd.read_pickle(fpath)

    if (data.index.names[-1] == 'permno') and (data.index.dtypes.iloc[-1] == 'int64'):
        data.index = data.index.set_levels(data.index.levels[-1].astype('Int64'), level=-1)

    if typecast:
        for col in data:
            if is_float(data[col]) and (data[col].dtype != config.float_type):
                data[col] = data[col].astype(config.float_type)
            elif data[col].dtype == 'object':
                if is_bool_array(data[col]):
                    data[col] = data[col].fillna(False).astype(bool)
                else:
                    data[col] = data[col].astype('string', errors='ignore')

    return data

