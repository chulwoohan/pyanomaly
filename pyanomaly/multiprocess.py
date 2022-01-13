import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyanomaly.globals import *

def process(i, module, fcn, data, *args, **kwargs):
    log(f'process {i} started.')
    kwargs['mp'] = False
    s = fcn.split('.')
    exec(f'from {module} import {s[0]}')  # import the relevant class or function
    return i, eval(fcn)(data, *args, **kwargs)

def multiprocess(func):
    max_prc = 10

    def wrapper(data, *args, **kwargs):
        if 'mp' in kwargs and not kwargs['mp']:
            kwargs.pop('mp')
            return func(data, *args, **kwargs)

        firms_list = np.array_split(data.index.get_level_values(-1).unique(), max_prc * 4)

        date_col = data.index.names[0]
        data = data.reset_index(level=0)
        char_list = [0] * len(firms_list)
        futures = []
        with ProcessPoolExecutor(max_workers=max_prc) as executor:
            for i, firms in enumerate(firms_list):
                data_ = data.loc[firms].set_index(date_col, append=True).swaplevel()
                futures.append(executor.submit(process, i, func.__module__, func.__qualname__, data_, *args, **kwargs))

            for f in as_completed(futures):
                i, char = f.result()
                char_list[i] = char
                log(f'Multiprocessing complete: {i}/{len(firms_list)}.')

        if isinstance(char_list[0], tuple):  # multiple return values
            n_var = len(char_list[0])
            retval = [[] for _ in range(n_var)]
            for i in range(n_var):
                for char in char_list:
                    retval[i].append(char[i])
                retval[i] = np.concatenate(retval[i])
        elif isinstance(char_list[0], np.ndarray):  # single return value (np.array)
            retval = np.concatenate(char_list)
        elif isinstance(char_list[0], pd.DataFrame):  # single return value (pd.DataFrame)
            retval = pd.concat(char_list)

        elapsed_time('Multiprocessing: Results gathered.')
        return retval

    return wrapper


