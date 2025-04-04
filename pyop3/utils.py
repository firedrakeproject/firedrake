from __future__ import annotations

import abc
import collections
import functools
import itertools
import numbers
import warnings
from collections.abc import Mapping
from typing import Any, Collection, Hashable, Optional

import numpy as np
import pytools
from immutabledict import ImmutableOrderedDict
from pyrsistent import pmap

from pyop3.config import config
from pyop3.exceptions import Pyop3Exception
from pyop3.dtypes import IntType

from mpi4py import MPI


class UnorderedCollectionException(Pyop3Exception):
    """Exception raised when an ordered collection is required."""


class EmptyCollectionException(Pyop3Exception):
    """Exception raised when a non-empty collection is required."""


class UniqueNameGenerator(pytools.UniqueNameGenerator):
    """Class for generating unique names."""

    def __call__(self, prefix: str) -> str:
        # To skip using prefix as a unique name we declare it as already used
        self.add_name(prefix, conflicting_ok=True)
        return super().__call__(prefix)


_unique_name_generator = UniqueNameGenerator()
"""Generator for creating globally unique names."""


def unique_name(prefix: str) -> str:
    return _unique_name_generator(prefix)


def maybe_generate_name(name, prefix, default_prefix, *, generator=_unique_name_generator):
    if name is not None:
        if prefix is not None:
            raise ValueError("Can only specify one of 'name' and 'prefix'")
        else:
            return name
    else:
        if prefix is not None:
            return generator(prefix)
        else:
            return generator(default_prefix)


# NOTE: Python 3.13 has warnings.deprecated
def deprecated(prefer=None, internal=False):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            msg = f"{fn.__qualname__} is deprecated and will be removed"
            if prefer:
                msg += f", please use {prefer} instead"
            warning_type = DeprecationWarning if internal else FutureWarning
            warnings.warn(msg, warning_type)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


class auto:
    pass


# type aliases
Id = str
Label = str


class Identified(abc.ABC):
    def __init__(self, id):
        self.id = id if id is not None else self.unique_id()

    @classmethod
    def unique_id(cls) -> str:
        return unique_name(f"_id_{cls.__name__}")


class Labelled(abc.ABC):
    def __init__(self, label):
        self.label = label if label is not None else self.unique_label()

    @classmethod
    def unique_label(cls) -> str:
        return unique_name(f"_label_{cls.__name__}")


# TODO is Identified really useful?
class UniqueRecord(pytools.ImmutableRecord, Identified):
    fields = {"id"}

    def __init__(self, id=None):
        pytools.ImmutableRecord.__init__(self)
        Identified.__init__(self, id)


# class Parameter(Identified):
#     """Wrapper class for a scalar value that differs between ranks."""
#     def __init__(self, value):
#         super().__init__(id=None)  # generate a fresh ID
#         self.box = np.array([value])
#
#     @property
#     def value(self):
#         return just_one(self.box)


# TODO: This is like Dat etc, a legit data carrier type
class Parameter:
    """Wrapper class for a scalar value that differs between ranks."""
    DEFAULT_PREFIX = "param"

    def __init__(self, value, *, name=None, prefix=None):
        self.value = as_numpy_scalar(value)
        self.name = maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

    @property
    def dtype(self):
        return self.value.dtype


class ValueMismatchException(Pyop3Exception):
    pass


class StrictlyUniqueDict(dict):
    """A dictionary where overwriting entries will raise an error."""

    def __setitem__(self, key, value, /) -> None:
        if key in self and value != self[key]:
            raise ValueMismatchException
        return super().__setitem__(key, value)

    # def update(self, other) -> None:
    #     shared_keys = self.keys() & other.keys()
    #     if len(shared_keys) > 0:
    #         raise ValueMismatchException
    #     super().update(other)



class OrderedSet(collections.abc.Sequence):
    """A mutable ordered set."""

    def __init__(self, values=None, /) -> None:
        # Python dicts are ordered so we use one to keep the ordering
        # and also have O(1) access.
        # self._values = {}

        # actually sometimes we have non-hashable things (PETSc Mats), so use a list
        # NOTE: This is very unsatisfying
        if values is not None:
            self._values = list(values)
        else:
            self._values = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._values!r})"

    def __str__(self) -> str:
        return f"{{{', '.join(self._values)}}})"

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

    def __or__(self, other, /) -> OrderedSet:
        # NOTE: other must be iterable
        merged = self.copy()
        for item in other:
            merged.add(item)
        return merged

    def index(self, value) -> int:
        return self._values.index(value)

    def count(self, value) -> int:
        return 1

    def copy(self) -> OrderedSet:
        return OrderedSet(self._values)

    def add(self, value):
        # self._values[value] = None
        if value not in self._values:
            self._values.append(value)


def as_tuple(item):
    if isinstance(item, collections.abc.Sequence):
        return tuple(item)
    else:
        return (item,)


def split_at(iterable, index):
    return iterable[:index], iterable[index:]


class PrettyTuple(tuple):
    """Implement a tuple with nice syntax for recursive functions. Like set notation."""

    def __or__(self, other):
        return type(self)(self + (other,))


class LengthMismatchException(Pyop3Exception):
    pass


@deprecated("Use zip(strict=True) instead")
def strict_zip(*iterables):
    return zip(*iterables, strict=True)


# old alias, remove
checked_zip = strict_zip


def rzip(*iterables):
    if any(not isinstance(it, collections.abc.Sized) for it in iterables):
        raise ValueError("Can only rzip with objects that have a known length")

    max_length = max(len(it) for it in iterables)
    return zip(*(pad(it, max_length, False) for it in iterables))


def pad(iterable, length, after=True, padding_value=None):
    missing = [padding_value] * (length - len(iterable))
    if after:
        return itertools.chain(iterable, missing)
    else:
        return itertools.chain(missing, iterable)


single_valued = pytools.single_valued
is_single_valued = pytools.is_single_valued


def merge_dicts(dicts, *, ordered=False):
    merged = {}
    for dict_ in dicts:
        merged.update(dict_)

    mapping_type = ImmutableOrderedDict if ordered else pmap
    return mapping_type(merged)


def unique(iterable):
    unique_items = []
    for item in iterable:
        if item not in unique_items:
            unique_items.append(item)
    return tuple(unique_items)


def has_unique_entries(iterable):
    # duplicate the iterator in case it can only be iterated over once (e.g. a generator)
    it1, it2 = itertools.tee(iterable, 2)
    return len(unique(it1)) == len(list(it2))


def is_sequence(item):
    return isinstance(item, collections.abc.Sequence)


def flatten(iterable):
    """Recursively flatten a nested iterable."""
    if isinstance(iterable, np.ndarray):
        return iterable.flatten()
    if not isinstance(iterable, (list, tuple)):
        return (iterable,)
    return tuple(item_ for item in iterable for item_ in flatten(item))


def some_but_not_all(iterable):
    # duplicate the iterable in case using any/all consumes it
    it1, it2 = itertools.tee(iterable)
    return any(it1) and not all(it2)


def strictly_all(iterable):
    """Returns ``all(iterable)`` but raises an exception if values are inconsistent."""
    if not isinstance(iterable, collections.abc.Iterable):
        raise TypeError("Expecting an iterable")

    # duplicate the iterable in case using any/all consumes it
    it1, it2 = itertools.tee(iterable)
    if (result := any(it1)) and not all(it2):
        raise ValueError("Iterable contains inconsistent values")
    return result


def just_one(iterable):
    # bit of a hack
    iterable = list(iterable)

    if len(iterable) == 0:
        raise ValueError("Empty iterable found")
    if len(iterable) > 1:
        raise ValueError("Too many values")
    return iterable[0]


class MultiStack:
    """Keyed stack."""

    def __init__(self, data=None):
        raise NotImplementedError("shouldnt be needed")
        self._data = data or collections.defaultdict(PrettyTuple)

    def __str__(self):
        return str(dict(self._data))

    def __repr__(self):
        return f"{self.__class__}({self._data!r})"

    def __getitem__(self, key):
        return self._data[key]

    def __or__(self, other):
        new_data = self._data.copy()
        if isinstance(other, collections.abc.Mapping):
            for key, value in other.items():
                new_data[key] += value
            return type(self)(new_data)
        else:
            return NotImplemented


def popwhen(predicate, iterable):
    """Pop the first instance from iterable where predicate is ``True``."""
    if not isinstance(iterable, list):
        raise TypeError("Expecting iterable to be a list")

    for i, item in enumerate(iterable):
        if predicate(item):
            return iterable.pop(i)
    raise KeyError("Predicate does not hold for any items in iterable")


# NOTE: It might be more logical for drop_last to default to True
def steps(sizes, *, drop_last=False):
    steps_ = np.concatenate([[0], np.cumsum(sizes)])
    return readonly(steps_[:-1]) if drop_last else readonly(steps_)


def pairwise(iterable):
    return zip(iterable, iterable[1:])


# stolen from stackoverflow
# https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
def invert(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p)  # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def invert_mapping(mapping, *, mapping_type=dict):
    return mapping_type((v, k) for k, v in mapping.items())


@functools.singledispatch
def strict_cast(obj: Any, dtype: type | np.dtype) -> Any:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@strict_cast.register(numbers.Integral)
def _(num: numbers.Integral, dtype: type | np.dtype) -> np.number:
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    iinfo = np.iinfo(dtype)
    if not (iinfo.min <= num <= iinfo.max):
        raise TypeError(f"{num} exceeds the limits of {dtype}")
    return dtype.type(num)


@strict_cast.register(np.ndarray)
def _(array: np.ndarray, dtype: type) -> np.ndarray:
    return array.astype(dtype, casting="safe")


def strict_int(num) -> IntType:
    return strict_cast(num, IntType)


def apply_at(func, iterable, index):
    if index < 0 or index >= len(iterable):
        raise IndexError

    result = []
    for i, item in enumerate(iterable):
        if i == index:
            result.append(func(item))
        else:
            result.append(item)
    return tuple(result)


def map_when(func, when_func, iterable):
    for item in iterable:
        if when_func(item):
            yield func(item)
        else:
            yield item


def readonly(array):
    """Return a readonly view of a numpy array."""
    view = array.view()
    view.setflags(write=False)
    return view


def debug_assert(predicate, msg=None):
    if config["debug"]:
        if msg:
            assert predicate(), msg
        else:
            assert predicate()


_ordered_mapping_types = (dict, collections.OrderedDict, ImmutableOrderedDict)


def is_ordered_mapping(obj: Mapping):
    return isinstance(obj, _ordered_mapping_types)


def expand_collection_of_iterables(compressed, /, *, ordered: bool = True) -> tuple:
    """
    Expand target paths written in 'compressed' form like:

        {key1: [item1, item2], key2: [item3]}

    Instead to the 'expanded' form:

        ({key1: item1, key2: item3}, {key1: item2, key2: item3})

    Valid input types for ``compressed`` include ordered mappings and iterables
    of 2-tuples (i.e. things that can be parsed into a `dict`).

    """
    # If `compressed` is not already a mapping then parse it to one
    if not isinstance(compressed, Mapping):
        compressed = dict(compressed)

    if ordered and not is_ordered_mapping(compressed):
        raise UnorderedCollectionException(
            "Expected an ordered mapping, valid options include: "
            f"{{{', '.join(type_.__name__ for type_ in _ordered_mapping_types)}}}"
        )

    mapping_type = ImmutableOrderedDict if ordered else pmap

    if not compressed:
        return (mapping_type(),)
    else:
        compressed_mut = dict(compressed)
        return _expand_dict_of_iterables_rec(compressed_mut, mapping_type=mapping_type)


def _expand_dict_of_iterables_rec(compressed_mut, /, *, mapping_type):
    expanded = []
    key, items = popfirst(compressed_mut)

    if compressed_mut:
        subexpanded = _expand_dict_of_iterables_rec(compressed_mut, mapping_type=mapping_type)
        for item in items:
            entry = mapping_type({key: item})
            for subentry in subexpanded:
                expanded.append(entry | subentry)
    else:
        for item in items:
            entry = mapping_type({key: item})
            expanded.append(entry)

    return tuple(expanded)


def popfirst(dict_: dict) -> Any:
    """Remove the first item from a dictionary and return it with its key."""
    if not dict_:
        raise EmptyCollectionException("Expected a non-empty dict")

    key = next(iter(dict_))
    return (key, dict_.pop(key))


class Record(abc.ABC):
    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and all(getattr(self, field) == getattr(other, field) for field in self._record_fields)
        )

    @property
    @abc.abstractmethod
    def _record_fields(self) -> frozenset:
        pass

    def _record_init(self, **kwargs) -> None:
        for field in self._record_fields:
            setattr(self, field, kwargs.pop(field))
        assert not kwargs

    def reconstruct(self, **kwargs):
        for field in self._record_fields:
            if field not in kwargs:
                kwargs[field] = getattr(self, field)

        new = object.__new__(type(self))
        new._record_init(**kwargs)
        return new


def unique_comm(iterable) -> MPI.Comm | None:
    comm = None
    for item in iterable:
        if not item.comm:
            continue

        if comm is None:
            comm = item.comm
        elif item.comm != comm:
            raise ValueError("More than a single comm provided")
    return comm


def as_numpy_scalar(value: numbers.Number) -> np.number:
    return just_one(np.asarray([value]))
