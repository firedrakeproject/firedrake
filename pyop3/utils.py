from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import itertools
import numbers
import operator
import warnings
from collections.abc import Callable, Iterable, Mapping, Hashable, Collection
from typing import Any

import numpy as np
import pytools
from immutabledict import immutabledict

from pyop3.config import config
from pyop3.exceptions import Pyop3Exception
from pyop3.dtypes import DTypeT, IntType

from mpi4py import MPI

import pyop3.extras.debug


# NOTE: Perhaps better inside another module
PYOP3_DECIDE = object()
"""Placeholder indicating that a value should be set by pyop3.

This is important in cases where the more traditional `None` is actually
meaningful.

"""


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
# class UniqueRecord(pytools.ImmutableRecord, Identified):
#     fields = {"id"}
#
#     def __init__(self, id=None):
#         pytools.ImmutableRecord.__init__(self)
#         Identified.__init__(self, id)


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


def merge_dicts(dicts: Iterable[Mapping], *, allow_duplicates=False) -> immutabledict:
    merged = {}
    for dict_ in dicts:
        merged.update(dict_)
    if not allow_duplicates and len(merged) != sum(map(len, dicts)):
        pyop3.extras.debug.warn_todo("duplicates found, this will become a hard error")
        # raise ValueError("Duplicates found")
    return immutabledict(merged)


def unique(iterable) -> tuple[Any]:
    unique_items = []
    for item in iterable:
        if item not in unique_items:
            unique_items.append(item)
    return tuple(unique_items)


def has_unique_entries(iterable):
    # duplicate the iterator in case it can only be iterated over once (e.g. a generator)
    it1, it2 = itertools.tee(iterable, 2)
    return len(unique(it1)) == len(list(it2))


def reduce(func, *args, **kwargs):
    if isinstance(func, str):
        match func:
            case "+":
                func = operator.add
            case "*":
                func = operator.mul
            case "|":
                func = operator.or_
            case _:
                raise ValueError

    return functools.reduce(func, *args, **kwargs)


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
    iterator = iter(iterable)

    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Empty iterable found")

    try:
        second = next(iterator)
    except StopIteration:
        return first

    raise ValueError("Too many values")


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


def steps(sizes, *, drop_last=True):
    steps_ = np.concatenate([[0], np.cumsum(sizes)])
    return readonly(steps_[:-1]) if drop_last else readonly(steps_)


def strides(sizes, *, drop_last=True) -> np.ndarray[int]:
    """
    Examples
    --------

    # I think...
    (2, 2) returns (2, 2) - 2i + j
    (1, 2) returns (2, 1) - 2i + j
    (2, 1) returns (1, 1) - i + j

    """
    assert drop_last, "old code otherwise"
    reversed_sizes = np.asarray(sizes, dtype=int)[::-1]
    strides_ = np.concatenate([[1], np.cumprod(reversed_sizes[:-1])], dtype=int)
    return readonly(strides_[::-1])


_nothing = object()
"""Sentinel value indicating nothing should be done.

This is useful in cases where `None` holds some meaning.

"""


def pairwise(iterable, *, final=_nothing):
    if final is not _nothing:
        return itertools.zip_longest(iterable, iterable[1:], fillvalue=final)
    else:
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
def _(num: numbers.Integral, dtype: DTypeT) -> np.number:
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    iinfo = np.iinfo(dtype)
    if not (iinfo.min <= num <= iinfo.max):
        raise TypeError(f"{num} exceeds the limits of {dtype}")
    return dtype.type(num)


@strict_cast.register(np.ndarray)
def _(array: np.ndarray, dtype: DTypeT) -> np.ndarray:
    return array.astype(dtype, casting="safe")


def strict_int(num: numbers.Number) -> IntType:
    return strict_cast(num, IntType)


def as_dtype(dtype: DTypeT | None, default: np.dtype) -> np.dtype:
    return np.dtype(dtype) if dtype else default


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


_ordered_mapping_types = (dict, collections.OrderedDict, immutabledict)


def is_ordered_mapping(obj: Mapping):
    return isinstance(obj, _ordered_mapping_types)


def expand_collection_of_iterables(compressed) -> tuple:
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

    if not compressed:
        return (immutabledict(),)
    else:
        compressed_mut = dict(compressed)
        return _expand_dict_of_iterables_rec(compressed_mut)


def _expand_dict_of_iterables_rec(compressed_mut):
    expanded = []
    key, items = popfirst(compressed_mut)

    if compressed_mut:
        subexpanded = _expand_dict_of_iterables_rec(compressed_mut)
        for item in items:
            entry = immutabledict({key: item})
            for subentry in subexpanded:
                expanded.append(entry | subentry)
    else:
        for item in items:
            entry = immutabledict({key: item})
            expanded.append(entry)

    return tuple(expanded)


def popfirst(dict_: dict) -> Any:
    """Remove the first item from a dictionary and return it with its key."""
    if not dict_:
        raise EmptyCollectionException("Expected a non-empty dict")

    key = next(iter(dict_))
    return (key, dict_.pop(key))


def record():
    return _make_record(eq=False)


def frozenrecord():
    return _make_record(frozen=True)


def _make_record(**kwargs):
    def wrapper(cls):
        cls = dataclasses.dataclass(**kwargs)(cls)
        cls.__record_init__ = _record_init
        return cls
    return wrapper


def _record_init(self: Any, **attrs: Mapping[str,Any]) -> Any:
    new = object.__new__(type(self))
    for field in dataclasses.fields(self):
        attr = attrs.pop(field.name, getattr(self, field.name))
        object.__setattr__(new, field.name, attr)

    if attrs:
        raise ValueError(
            f"Unrecognised arguments encountered during initialisation: {', '.join(attrs)}"
        )

    if hasattr(new, "__post_init__"):
        new.__post_init__()

    return new


def attr(attr_name: str) -> property:
    return property(lambda self: getattr(self, attr_name))


@functools.singledispatch
def freeze(obj: Any) -> Hashable:
    raise TypeError


@freeze.register
def _(tuple_: tuple) -> tuple:
    return tuple(map(freeze, tuple_))


@freeze.register
def _(list_: list) -> tuple:
    return tuple(map(freeze, list_))


@freeze.register
def _(immutabledict_: immutabledict) -> immutabledict:
    return immutabledict({
        key: freeze(value)
        for key, value in immutabledict_.items()
    })


@freeze.register
def _(dict_: dict) -> immutabledict:
    return immutabledict({
        key: freeze(value)
        for key, value in dict_.items()
    })


@freeze.register
def _(hashable: Hashable) -> Hashable:
    return hashable


def unique_comm(iterable) -> MPI.Comm | None:
    if isinstance(iterable, np.ndarray):
        iterable = iterable.flatten()

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


def filter_type(type_: type, iterable: Iterable):
    return filter(lambda item: isinstance(item, type_), iterable)


def ceildiv(a, b, /):
    # See https://stackoverflow.com/a/17511341
    return -(a // -b)


