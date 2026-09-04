import collections
import itertools
from collections.abc import Hashable

import pyop3.collections


class Renamer:
    """Class for renaming things inside a visitor.

    The class support two types of renaming: by type and by instance. This is
    because sometimes instance renaming is insufficient. For example you may
    have the axis label to hand but not the actual axis instance.

    Note that each rename map is assumed to be distinct. No checking is
    performed to see if there are clashes.

    """

    def __init__(
        self,
        *,
        existing_type_store=None,
        existing_obj_store=None,
        allow_missing: bool = True,
    ) -> None:
        type_store = collections.defaultdict(pyop3.collections.StrictlyUniqueDict)
        if existing_type_store:
            for type_, names in existing_type_store.items():
                type_store[type_] |= names

        obj_store = pyop3.collections.StrictlyUniqueDict()
        if existing_obj_store:
            for obj, name in existing_obj_store.items():
                obj_store[obj] = name

        self.allow_missing = allow_missing
        self.type_store = type_store
        self._counter_by_type = collections.defaultdict(itertools.count)
        self.obj_store = obj_store
        self._counter_by_obj_type = collections.defaultdict(itertools.count)

    def add_type(self, type_: type, key: Hashable) -> str:
        try:
            return self.type_store[type_][key]
        except KeyError as err:
            if not self.allow_missing:
                raise err

        existing_names = self._all_names
        while True:
            num = next(self._counter_by_type[type_])
            name = f"{type_.__name__}_{num}"
            if name not in existing_names:
                break
        self.type_store[type_][key] = name
        return name

    def add_obj(self, obj: Hashable) -> str:
        try:
            return self.obj_store[obj]
        except KeyError:
            if not self.allow_missing:
                raise err

        existing_names = self._all_names
        while True:
            num = next(self._counter_by_obj_type[type(obj)])
            name = f"{type(obj).__name__}_{num}"
            if name not in existing_names:
                break
        self.obj_store[obj] = name
        return name

    @property
    def _all_names(self) -> set:
        all_names = set()
        for names in self.type_store.values():
            all_names.update(names.values())
        all_names.update(self.obj_store.values())
        return all_names
