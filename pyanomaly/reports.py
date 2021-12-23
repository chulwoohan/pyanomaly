import numpy as np
import pandas as pd

from .globals import *


# def table_regression(mean, tval, axis=0):
#     if axis == 0: # Series. mean[0], tval[0], mean[1], tval[1], ...
#         tval = tval.rename({i: '  t-stat' for i in tval.index})
#         tab = pd.concat([mean.iloc[:, 0], tval.iloc[:, 0]])
#         idx = []
#         for i in range(len(mean)):
#             idx.append(i)
#             idx.append(len(mean) + i)
#
#         return tab.iloc[idx]
#     else:  # DataFrame. columns = ['mean', 't-stat']
#         return pd.concat([mean, tval], axis=1)

