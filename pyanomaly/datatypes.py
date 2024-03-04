"""This module defines data types.

    * `struct`: Dict that can access items using '.' expression or integer indexing.
"""

class struct(dict):
    """Dict that can access items using '.' expression or integer indexing.

    Examples:
        >>> x = struct(
        ...    a='py',
        ...    b='anomaly'
        ...    )
        >>> x['a']
        py
        >>> x.a
        py
        >>> x[0]
        py
    """

    def __init__(self, *args, **kwargs):
        if len(args) and type(args[0]) == dict:
            self.update(args[0])
        else:
            super().__init__(*args, **kwargs)
        for k, v in self.items():
            self.__dict__[k] = v

    def __setitem__(self, k, value):
        # struct['x'] = y
        self.__dict__[k] = value
        super().__setitem__(k, value)

    def __setattr__(self, k, value):
        # struct.x = y
        self.__dict__[k] = value
        if k in self.keys():  # update dict only when the key already exists.
            super().__setitem__(k, value)

    def __getitem__(self, k):
        if type(k) == str:
            return super().__getitem__(k)
        else:  # int
            return list(self.values())[k]

    def __copy__(self):
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.update(self)
        cpy.__dict__.update(self.__dict__)
        return cpy

    def copy(self):
        return self.__copy__()

