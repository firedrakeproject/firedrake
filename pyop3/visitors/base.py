import collections
import itertools
from collections.abc import Mapping
from typing import Hashable


class Renamer:
    def __init__(
        self,
        rename_map: Mapping[type | None, Mapping[str, str]] | None = None,
    ) -> None:
        if rename_map is None:
            rename_map = {}
        self.rename_map = rename_map
        self._store = collections.defaultdict(dict)
        self._counter_by_type = collections.defaultdict(itertools.count)

    def get(self, key: Hashable, *, type_: type | None = None) -> str:
        assert not isinstance(key, tuple), "old api"

        if type_ is None:
            type_ = type(key)

        if type_ in self.rename_map:
            return self.rename_map[type_][key]

        return self._store[type_][key]

    def add(self, key: Hashable, *, type_: type | None = None) -> str:
        assert not isinstance(key, tuple), "old api"
        if type_ is None:
            type_ = type(key)


        if type_ in self.rename_map:
            return self.rename_map[type_][key]

        try:
            return self.get(key, type_=type_)
        except KeyError:
            index = next(self._counter_by_type[type_])
            label = f"{type_.__name__}_{index}"
            self._store[type_][key] = label
            return self.get(key, type_=type_)
