import collections
import itertools
from collections.abc import Mapping
from typing import Hashable


class Renamer:
    """Class for renaming things inside a visitor.

    The class support two types of renaming: by type and by instance. This is
    because sometimes instance renaming is insufficient. For example you may
    have the axis label to hand but not the actual axis instance.

    Note that each rename map is assumed to be distinct. No checking is
    performed to see if there are clashes.

    """

    def __init__(self) -> None:
        self._type_store = collections.defaultdict(dict)
        self._counter_by_type = collections.defaultdict(itertools.count)
        self._obj_store = {}
        self._counter_by_obj_type = collections.defaultdict(itertools.count)

    def add_type(self, type_: type, key: Hashable) -> str:
        try:
            return self._type_store[type_][key]
        except KeyError:
            num = next(self._counter_by_type[type_])
            name = f"{type_.__name__}_{num}"
            self._type_store[type_][key] = name
            return name

    def add_obj(self, obj: Hashable) -> str:
        try:
            return self._obj_store[obj]
        except KeyError:
            num = next(self._counter_by_obj_type[type(obj)])
            name = f"{type(obj).__name__}_{num}"
            self._obj_store[obj] = name
            return name
