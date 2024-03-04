"""This module defines utility functions.

    .. autosummary::
        :nosignatures:

        is_iterable
        to_list
        unique_list
        drop_columns
        keep_columns
        is_zero
        is_float
        is_numeric
        is_bool_array
        nansum1
"""

from collections.abc import Iterable
import numpy as np
import pandas as pd


def is_iterable(x):
    """Check if a variable is iterable.

    Check if `x` is iterable. A string, an iterable object, is considered not iterable.

    Args:
        x: A variable to check.

    Returns:
         Bool. True if `x` is iterable.
    """

    return (not isinstance(x, str)) and isinstance(x, Iterable)


def to_list(x):
    """Convert a variable to a list.

    Args:
        x: A scalar or an iterable item.

    Returns:
        `x` converted to list.

    Examples:
        >>> x = 1
        ... to_list(x)
        [1]
        >>> x = [1, 2]
        ... to_list(x)
        [1, 2]
        >>> x = (1, 2)
        ... to_list(x)
        [1, 2]
    """

    if is_iterable(x):
        return list(x)
    else:
        return [x]


def unique_list(x, exclude_nan=True):
    """Get a list of unique elements of a list.

    If `x` contains a list, its elements are considered elements of `x`, not the list itself.

    Args:
        x: A list.
        exclude_nan: If True, exclude None and np.nan elements.

    Returns:
        List of unique elements of `x`.

    Examples:
        >>> x = [1, 2, 2, None]
        ... unique_list(x)
        [1, 2]
        >>> x = [1, 2, [1, 3], None]
        ... unique_list(x)
        [1, 2, 3]
    """

    y = []
    for i in x:
        if is_iterable(i):
            for j in i:
                y.append(j)
        else:
            y.append(i)

    y = list(set(y))
    if exclude_nan:
        notna = pd.notna(y)
        y = [v for i, v in enumerate(y) if notna[i]]
    return y


import gc
def drop_columns(data, cols):
    """Delete columns of a DataFrame.

    Columns are deleted in-place.

    Args:
        data: DataFrame.
        cols: List of columns to drop.
    """

    for col in cols:
        del data[col]


def keep_columns(data, cols):
    """Keep columns of a DataFrame.

    Keep `cols` columns of `data` and delete the rest in-place.
    Much more memory-efficient than the following two methods: these methods momentarily consume a lot of memory when
    `data` is large.

    >>> data = data[cols]
    >>> data.drop(columns=data.columns.difference(cols), inplace=True)

    Use this function when handling a large dataset.

    Args:
        data: DataFrame.
        cols: List of columns to keep.
    """

    del_cols = data.columns.difference(cols)
    drop_columns(data, del_cols)


ZERO = 1.e-7


def is_zero(x):
    """Check if a variable or its element is zero.

    A value is considered 0 if it is in the range (-1.e-7, 1.e-7).

    Args:
        x: A scalar or array.

    Returns:
        Bool or an array of bool. True if 0.
    """

    return (x < ZERO) & (x > -ZERO)


def is_float(x):
    """Check if a variable's data type is float.

    Args:
        x: A scalar or array.

    Returns:
        Bool. True if the data type of `x` is float.
    """

    try:
        dtype = x.dtype
    except:
        dtype = type(x)

    return (dtype == float) | (dtype == np.float64) | (dtype == np.float32)


def is_int(x):
    """Check if a variable's data type is int.

    Args:
        x: A scalar or array.

    Returns:
        Bool. True if the data type of `x` is int.
    """

    try:
        dtype = x.dtype
    except:
        dtype = type(x)

    return (dtype == int) | (dtype == np.int64) | (dtype == np.int32)


def is_numeric(x):
    """Check if a variable's data type is numeric (int or float).

    Args:
        x: A scalar or array.

    Returns:
        Bool. True if the data type of `x` is numeric.
    """

    return is_float(x) | is_int(x)


def is_bool_array(array):
    """Check if an array is a bool array.

    The `array` is identified as a bool array if:

        * its dtype is 'bool' or 'boolean'; or
        * it contains only True, False, or None.

    Args:
        array: Series or ndarray.

    Returns:
        True if `array` is a bool array.
    """

    if array.dtype == 'bool':
        return True
    else:
        unique_values = pd.unique(array)
        notna = pd.notna(unique_values)
        unique_values = [v for i, v in enumerate(unique_values) if notna[i]]

        return unique_values in ([True], [False], [True, False], [False, True])


def nansum1(*args):
    """Summation treating nan values as zero.

    This is similar to ``sum()`` of SAS: nan's of `args` are replaced by 0. If all elements are nan, the result is nan.

    Args:
        args: List of Series.

    Returns:
        Series. Sum of `args`

    Examples:
        >>> x = pd.Series([np.nan, 1, 1])
        ... y = pd.Series([np.nan, np.nan, 1])

        >>> nansum(x, y)
        0     NaN
        1   1.000
        2   2.000
        dtype: float64

        >>> nansum(x, y, y)
        0     NaN
        1   1.000
        2   3.000
    """

    y = args[0].fillna(0)
    isnan = args[0].isna()
    for x in args[1:]:
        y += x.fillna(0)
        isnan &= x.isna()
    y[isnan] = np.nan

    return y
