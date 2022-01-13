"""This module defines functions for file IO."""

from .globals import *

def write_to_file(data, fname, fdir=None):
    """Write `data` to a pickle file.

    Args:
        data (Dataframe): Data to save.
        fname: File name without extension.
        fdir: Directory. If None, `config.output_dir`.

    """
    fdir = fdir or config.output_dir

    if not os.path.isdir(fdir):
        os.mkdir(fdir)

    data.to_pickle(fdir + ('' if fdir[-1] == '/' else '/') + fname + '.pickle')


def read_from_file(fname, fdir=None):
    """Read data from a pickle file.

    Args:
        fname: File name without extension.
        fdir: Directory. If None, `config.output_dir`.

    Returns:
        DataFrame read from `fdir/fname.pickle`.
    """
    fdir = fdir or config.output_dir
    fpath = fdir + ('' if fdir[-1] == '/' else '/') + fname + '.pickle'
    return pd.read_pickle(fpath)



