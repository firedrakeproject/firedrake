from __future__ import annotations

import collections
import pprint

import numpy as np
from immutabledict import immutabledict as idict

from pyop3 import utils
from pyop3.exceptions import ValueMismatchException


class AlwaysEmptyDict(dict):
    def __init__(self) -> None:
        super().__init__()

    def __setitem__(self, key, value, /) -> None:
        pass

    def setdefault(self, key, default=None, /):
        return default


class StrictlyUniqueDict(dict):
    """A dictionary where overwriting entries will raise an error."""
    def __setitem__(self, key, value, /) -> None:
        if key in self and value != self[key]:
            raise ValueMismatchException
        return super().__setitem__(key, value)


class StrictlyUniqueDefaultDict(collections.defaultdict):
    def __setitem__(self, key, value, /) -> None:
        if key in self and value != self[key]:
            raise ValueMismatchException
        return super().__setitem__(key, value)


# NOTE: This has a lot of scope for improvements
class UniqueList(list):
    def append(self, value, /) -> None:
        if value in self:
            raise ValueMismatchException
        return super().append(value)


class AbstractOrderedSet:

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._values!r})"

    def __str__(self) -> str:
        return f"{{{', '.join(map(str, self._values))}}}"

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other, /) -> bool:
        return type(other) is type(self) and other._values == self._values

    def __getitem__(self, index, /):
        return self._values[index]

    def __contains__(self, item, /) -> bool:
        return item in self._values

    def __iter__(self):
        # return iter(self._values.keys())
        return iter(self._values)

    def __reversed__(self):
        return iter(reversed(self._values))

    def __or__(self, other, /) -> Self:
        assert is_ordered_sequence(other)
        values = list(self._values)
        for item in other:
            if item not in values:
                values.append(item)
        return type(self)(values)

    def union(self, /, *others) -> Self:
        new = self
        for other in others:
            new |= other
        return new

    def index(self, value, /) -> Any:
        return self._values.index(value)


class OrderedSet(AbstractOrderedSet):
    """A mutable ordered set."""

    def __init__(self, values=None, /) -> None:
        if values is None:
            values = []
        else:
            assert is_ordered_sequence(values) or len(values) < 2
            values = list(values)

        self._values = values

    def index(self, value) -> int:
        return self._values.index(value)

    def count(self, value) -> int:
        # why did I write this?
        raise NotImplementedError

    def copy(self) -> OrderedSet:
        return OrderedSet(self._values)

    def add(self, value):
        # self._values[value] = None
        if value not in self._values:
            self._values.append(value)

    def update(self, /, *others):
        for other in others:
            for item in other:
                self.add(item)

    def remove(self, value):
        try:
            index = self._values.index(value)
        except ValueError:
            raise KeyError
        else:
            self._values.pop(index)

    def clear(self):
        self._values.clear()


class OrderedFrozenSet(AbstractOrderedSet):

    def __init__(self, values: collections.abc.Sequence = (), /) -> None:
        assert is_ordered_sequence(values) or len(values) < 2
        self._values = utils.unique(values)

    def __hash__(self) -> int:
        return hash((type(self), self._values))


# monkey patch pretty printing
pprint.PrettyPrinter._dispatch[idict.__repr__] = pprint.PrettyPrinter._pprint_ordered_dict
pprint.PrettyPrinter._dispatch[OrderedSet.__repr__] = pprint.PrettyPrinter._pprint_set
pprint.PrettyPrinter._dispatch[OrderedFrozenSet.__repr__] = pprint.PrettyPrinter._pprint_set


_ordered_mapping_types = (dict, collections.OrderedDict, idict)

_dict_keys_type = type({}.keys())
_dict_values_type = type({}.values())
_dict_items_type = type({}.items())
_ordered_sequence_types = (
    list,
    tuple,
    _dict_keys_type,
    _dict_values_type,
    _dict_items_type,
    np.ndarray,
    AbstractOrderedSet,
)


def is_ordered_mapping(obj: Mapping) -> bool:
    return isinstance(obj, _ordered_mapping_types)


def is_ordered_sequence(obj: collections.abc.Sequence) -> bool:
    return isinstance(obj, _ordered_sequence_types)


