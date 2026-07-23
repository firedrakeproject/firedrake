import collections
import itertools
from typing import Hashable


class Renamer:
    def __init__(self):
        self.store = {}
        self._counter_by_type = collections.defaultdict(itertools.count)

    def __getitem__(self, key):
        return self.store[key]

    def add(self, key: tuple[type, Hashable] | Hashable) -> str:
        if isinstance(key, tuple):
            obj_type, _ = key
        else:
            obj_type = type(key)

        try:
            return self.store[key]
        except KeyError:
            index = next(self._counter_by_type[obj_type])
            label = f"{obj_type.__name__}_{index}"
            return self.store.setdefault(key, label)
