"""This module defines utility functions."""

from collections.abc import Iterable
from pyanomaly.globals import *


def is_iterable(x):
    """Check if `x` is iterable.

    A string, a iterable object, is considered not iterable in this function.

    Returns:
         True if `x` is iterable.
    """

    return (not isinstance(x, str)) and isinstance(x, Iterable)


def to_list(x):
    """Convert `x` to a list.

    Args:
        x: a list, tuple, or scalar.

    Returns:
        `x` converted to a list.
    """

    if is_iterable(x):
        return list(x)
    else:
        return [x]


def unique_list(x, exclude_nan=True):
    """Get a list of unique elements of `x`.

    If `x` contains a list, its elements are considered elements of `x`, not the list itself.

    Args:
        x: A list.
        exclude_nan: If True, exclude None elements.

    Returns:
        List of unique element of `x`.
    """

    y = []
    for i in x:
        if is_iterable(i):
            for j in i:
                y.append(j)
        else:
            y.append(i)

    y = list(set(y))
    if exclude_nan and None in y:
        y.remove(None)
    return y


def drop_columns(data, cols):
    """Delete `cols` columns from `data`."""

    for col in cols:
        del data[col]


def keep_columns(data, cols):
    """Delete columns of `data` except `cols`.

    Equivalent to `data = data[cols]`, but much more memory efficient.
    Not sure why but the following two methods seem to momentarily consume a lot of memory:

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
    """Check if `x` is zero.
    """

    return (x < ZERO) & (x > -ZERO)