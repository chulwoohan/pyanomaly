"""This module defines logging functions.

    .. autosummary::
        :nosignatures:

        set_log_path
        log
        err
        warn
        debug
        drawline
        start_timer
        elapsed_time
"""

import time as t
import sys
import os
import pandas as pd
from .config import get_config

log_path = None


def set_log_path(fpath=None, append=True):
    """Set log file path.

    Args:
        fpath: Log file name or path or ``__file__``. If it's a name, the path becomes ``config.log_dir`` + `fpath`.
            If ``__file__`` , the name of the module calling this function is retrieved from ``__file__`` and
            the path is set to ``config.log_dir`` + module name. If None, the path is ``config.log_dir`` + 'log.log'.
        append: If True, append to the existing log file. Otherwise, delete the current log file.

    Examples:

        >>> set_log_path('./log/example.log')  # Full file path.
        >>> set_log_path('example.log')  # file name. Path = config.log_dir + 'example.log'
        >>> set_log_path(__file__)  # Make path from a module name. Path = config.log_dir + module name
    """

    global log_path

    if not fpath:
        log_path = get_config('log_dir') + 'log.log'
    elif fpath[-3:] == '.py':  # __file__
        fname = fpath[fpath.rfind('/') + 1:]
        fname = fname[fname.rfind('\\') + 1:-3]
        log_path = get_config('log_dir') + fname + '.log'
    elif (fpath.rfind('/') == -1) and (fpath.rfind('\\') == -1):  # file name
        log_path = get_config('log_dir') + fpath
    else:  # full path
        log_path = fpath

    fdir = log_path[:max(log_path.rfind('/'), log_path.rfind('\\'))]
    if not os.path.isdir(fdir):
        os.mkdir(fdir)

    if not append:
        open(log_path, 'w')


def log(msg, headmsg=None, header=True):
    """Write log message.

    The format is [yyyy/mm/dd hh:mm: `headmsg`] `msg`.

    Args:
        msg: Log message.
        headmsg: Message header.
        header: If True, write the header.
    """

    if isinstance(msg, pd.DataFrame) or isinstance(msg, pd.Series):
        header = False
        msg = f'\n{msg}'

    if header and headmsg:
        head = f"[{t.strftime('%Y/%m/%d %H:%M', t.localtime())}: {headmsg}] "
    elif header:
        head = f"[{t.strftime('%Y/%m/%d %H:%M', t.localtime())}] "
    elif headmsg:
        head = f"[{headmsg}] "
    else:
        head = ""

    text = head + str(msg)
    print(text)
    if log_path and (log_path != sys.stdout):
        with open(log_path, 'a') as f:
            print(text, file=f)


def err(msg):
    """Write error message.

    Args:
        msg: Error message.
    """

    log(msg, 'ERROR')


def warn(msg):
    """Write warning message.

    Args:
        msg: Warning message.
    """

    log(msg, 'WARNING')


def debug(msg=''):
    """Write debugging message.

    Args:
        msg: Debugging message.
    """

    if get_config('debug'):
        elapsed_time(msg, 'DEBUG')


def drawline(level=0, width=80):
    """Draw line in the log file.

    Args:
        level: Shape of line. 0: '#', 1: '*', 2: '-'
        width: Line width.
    """

    if level == 0:
        log("#" * width)
    elif level == 1:
        log("*" * width)
    elif level == 2:
        log("-" * width)
    elif isinstance(level, str):
        log(level * width)


def start_timer(msg='', headmsg=None):
    """Start timer.

    Args:
        msg: Message to print.
        headmsg: Message header.
    """

    global tstart, tprev
    tstart = t.perf_counter()
    tprev = tstart
    log(f"Timer On: {msg}", headmsg)


def elapsed_time(msg='', headmsg=None):
    """Get elapsed time.

    The total elapsed time ('hh:mm:ss') since the timer started and the elapsed time (in seconds) since the last
    call of this function are printed. If :func:`start_timer` was never called before, this function starts timer.

    Args:
        msg: Message to print.
        headmsg: Message header.

    Examples:
        >>> start_timer('start')
        [2024/02/06 01:14] Timer On: start

        >>> elapsed_time('check1')
        [2024/02/06 01:15] Elapsed [0:00:35.564, 35.564]: check1

        >>> elapsed_time('check2', 'parallel1')
        [2024/02/06 01:15: parallel1] Elapsed [0:00:53.802, 18.238]: check1
    """

    try:
        global tstart, tprev
        tnow = t.perf_counter()
        m, s = divmod(tnow - tstart, 60)
        h, m = divmod(int(m), 60)
        log(f"Elapsed [{h:d}:{m:02d}:{s:06.3f}, {tnow - tprev:0.3f}]: {msg}", headmsg)
        tprev = tnow
    except:
        start_timer(msg, headmsg)
