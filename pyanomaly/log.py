import time as t
import sys
import os
import pandas as pd

log_path = None


def set_log_path(fpath):
    global log_path
    log_path = fpath

    fdir = fpath[:max(fpath.rfind('/'), fpath.rfind('\\'))]
    if not os.path.isdir(fdir):
        os.mkdir(fdir)


def log(msg, headmsg=None, header=True):
    if isinstance(msg, pd.DataFrame) or isinstance(msg, pd.Series):
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
    return log(msg, 'ERROR')


def warn(msg):
    return log(msg, 'WARNING')


def drawline(level=0, width=80):
    if level == 0:
        log("#" * width)
    elif level == 1:
        log("*" * width)
    elif level == 2:
        log("-" * width)
    elif isinstance(level, str):
        log(level * width)


def start_timer(msg='', test_id=None):
    global tstart, tprev, testid
    tstart = t.perf_counter()
    tprev = tstart
    testid = test_id
    log(f"Timer On: {msg}", testid)


def elapsed_time(msg=''):
    try:
        global tstart, tprev, testid
        tnow = t.perf_counter()
        m, s = divmod(int(tnow - tstart + 1), 60)
        h, m = divmod(m, 60)
        log(f"Elapsed [{h:d}:{m:02d}:{s:02d}, {tnow - tprev:0.3f}]: {msg}", testid)
        tprev = tnow
    except:
        start_timer(msg)
