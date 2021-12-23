
class struct(dict):
    """dict that allows '.' expression.
    Eg:
        x = struct(
            a='py',
            b='anomaly'
            )
        x['a'] and x.a both return 'py'.

    """
    def __init__(self, *args, **kwargs):
        if len(args) and type(args[0]) == dict:
            self.update(args[0])
        else:
            super().__init__(*args, **kwargs)
        for k, v in self.items():
            self.__dict__[k] = v

    def __setitem__(self, k, value):
        self.__dict__[k] = value
        super().__setitem__(k, value)

    def __setattr__(self, k, value):
        self.__dict__[k] = value
        if k in self.keys():  # update dict only when the key already exists.
            super().__setitem__(k, value)

    def __copy__(self):
        cls = self.__class__
        cpy = cls.__new__(cls)
        cpy.update(self)
        cpy.__dict__.update(self.__dict__)
        return cpy

    def copy(self):
        return self.__copy__()


class iterstruct(struct):
    """Iterable struct."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._idx = 0

    def __getitem__(self, k):
        if type(k) == str:
            return super().__getitem__(k)
        else:  # int
            return list(self.values())[k]

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx == len(self):
            self._idx = 0
            raise StopIteration
        else:
            self._idx += 1
            return list(self.values())[self._idx - 1]

