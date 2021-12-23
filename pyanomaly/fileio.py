from .globals import *

def write_to_file(data, fname, fdir=None):
    """Save data to a pickle file.

    Args:
        data: data to save (Dataframe)
        fname: file name
        fdir: directory. If None, config.output_dir.

    """
    fdir = fdir or config.output_dir

    if not os.path.isdir(fdir):
        os.mkdir(fdir)

    data.to_pickle(fdir + ('' if fdir[-1] == '/' else '/') + fname + '.pickle')


def read_from_file(fname, fdir=None):
    """Read data from file.

    Args:
        fname: file name
        fdir: directory

    Returns:
        data read.
    """
    fdir = fdir or config.output_dir
    fpath = fdir + ('' if fdir[-1] == '/' else '/') + fname + '.pickle'
    return pd.read_pickle(fpath)



