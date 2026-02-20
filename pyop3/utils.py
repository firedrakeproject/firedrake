from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import functools
import itertools
import numbers
import operator
import warnings
from collections.abc import Callable, Iterable, Mapping, Hashable, Collection
from typing import Any

import cachetools
import numpy as np
import pytools
from immutabledict import immutabledict
from mpi4py import MPI


from pyop3.collections import AbstractOrderedSet
from pyop3.config import config
from pyop3.constants import PYOP3_DECIDE, _nothing
from pyop3.dtypes import DTypeT, IntType
from pyop3.exceptions import CommMismatchException, CommNotFoundException, Pyop3Exception, UnhashableObjectException
from pyop3.mpi import collective


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


# remove me
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
        self.label = label if label is not PYOP3_DECIDE else self.unique_label()

    @classmethod
    def unique_label(cls) -> str:
        return unique_name(f"_label_{cls.__name__}")


def as_tuple(item: Any) -> tuple[Any, ...]:
    if isinstance(item, collections.abc.Iterable):
        return tuple(item)
    else:
        return (item,)


def split_at(iterable, index):
    return iterable[:index], iterable[index:]


single_valued = pytools.single_valued
is_single_valued = pytools.is_single_valued


def merge_dicts(dicts: Iterable[Mapping]) -> immutabledict:
    merged = StrictlyUniqueDict()
    for dict_ in dicts:
        merged.update(dict_)
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


def is_sorted(array: np.ndarray) -> np.bool:
    """
    Notes
    -----
    This function works even for empty arrays, which are reported as being
    sorted.

    """
    return np.all(array[:-1] <= array[1:])


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


def just_one(iterable: collections.abc.Iterable) -> Any:
    """Return the only entry in an iterable.

    Parameters
    ----------
    iterable
        The container with only a single entry.

    Returns
    -------
    Any
        The single item in ``iterable``.

    Raises
    ------
    ValueError
        If the iterable does not contain a single entry.

    """
    iterator = iter(iterable)

    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Iterable is empty")

    try:
        second = next(iterator)
    except StopIteration:
        return first
    else:
        raise ValueError("Too many values")


def popwhen(predicate, iterable):
    """Pop the first instance from iterable where predicate is ``True``."""
    if not isinstance(iterable, list):
        raise TypeError("Expecting iterable to be a list")

    for i, item in enumerate(iterable):
        if predicate(item):
            return iterable.pop(i)
    raise KeyError("Predicate does not hold for any items in iterable")


def steps(sizes, *, drop_last=True, dtype=None):
    if isinstance(sizes, np.ndarray):
        assert dtype is None
        dtype = sizes.dtype
    steps_ = np.concatenate([[0], np.cumsum(sizes)], dtype=dtype)
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
    # reversed_sizes = np.asarray(sizes, dtype=int)[::-1]
    # strides_ = np.concatenate([[1], np.cumprod(reversed_sizes[:-1])], dtype=int)
    reversed_sizes = np.asarray(sizes)[::-1]
    strides_ = np.concatenate([[1], np.cumprod(reversed_sizes[:-1])])
    return readonly(strides_[::-1])



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


def strict_cast(obj: Any, dtype: type | np.dtype) -> Any:
    if isinstance(obj, numbers.Number):
        return np.array([obj]).astype(dtype, casting="same_kind").item()
    else:
        return obj.astype(dtype, casting="same_kind")


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
    if config.debug_checks:
        if msg:
            assert predicate(), msg
        else:
            assert predicate()


_ordered_mapping_types = (dict, collections.OrderedDict, immutabledict)

_dict_keys_type = type({}.keys())
_dict_values_type = type({}.values())
_dict_items_type = type({}.items())
_ordered_sequence_types = (
    list,
    tuple,
    AbstractOrderedSet,
    _dict_keys_type,
    _dict_values_type,
    _dict_items_type,
)


def is_ordered_mapping(obj: Mapping) -> bool:
    return isinstance(obj, _ordered_mapping_types)


def is_ordered_sequence(obj: collections.abc.Sequence) -> bool:
    return isinstance(obj, _ordered_sequence_types)


# TODO: case for using typing generics
# TODO: signature is slightly wrong, can pass anything that can be cast to a dict
def expand_collection_of_iterables(compressed: Mapping[Hashable, Sequence[Any]]) -> tuple[idict[Hashable, Any], ...]:
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


def split_by(condition, items):
    """Split an iterable in two according to some condition.

    :arg condition: Callable applied to each item in ``items``, returning ``True``
        or ``False``.
    :arg items: Iterable to split apart.
    :returns: A 2-tuple of the form ``(yess, nos)``, where ``yess`` is a tuple containing
        the entries of ``items`` where ``condition`` is ``True`` and ``nos`` is a tuple
        of those where ``condition`` is ``False``.
    """
    result = [], []
    for item in items:
        if condition(item):
            result[0].append(item)
        else:
            result[1].append(item)
    return tuple(result[0]), tuple(result[1])




def popfirst(dict_: dict) -> Any:
    """Remove the first item from a dictionary and return it with its key."""
    if not dict_:
        raise EmptyCollectionException("Expected a non-empty dict")

    key = next(iter(dict_))
    return (key, dict_.pop(key))


@functools.singledispatch
def freeze(obj: Any) -> Hashable:
    raise UnhashableObjectException


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


def single_comm(objects, /, comm_attr: str, *, allow_undefined: bool = False) -> MPI.Comm | None:
    assert len(objects) > 0

    comm = None
    for item in iterflat(objects):
        item_comm = getattr(item, comm_attr, None)

        if item_comm is None:
            if allow_undefined:
                continue
            else:
                raise CommNotFoundException("Object does not have an associated communicator")

        if comm is None:
            comm = item_comm
        elif item_comm != comm:
            raise CommMismatchException("Multiple communicators found")
    return comm


@collective
def common_comm(objects, /, comm_attr: str, *, allow_undefined: bool = False) -> MPI.Comm | None:
    """Return a communicator valid for all objects.

    This is defined as the communicator with the largest size. I *think* that
    this is the right way to think about this.

    """
    assert len(objects) > 0

    selected_comm = None
    for item in iterflat(objects):
        item_comm = getattr(item, comm_attr, None)

        if item_comm is None:
            if allow_undefined:
                continue
            else:
                raise CommNotFoundException("Object does not have an associated communicator")

        if selected_comm is None or item_comm.size > selected_comm.size:
            selected_comm = item_comm
    assert selected_comm is not None
    return selected_comm


def iterflat(iterable):
    if isinstance(iterable, np.ndarray):
        iterable = iterable.flatten()
    return iter(iterable)


def as_numpy_scalar(value: numbers.Number) -> np.number:
    return just_one(np.asarray([value]))


def filter_type(type_: type, iterable: Iterable):
    return filter(lambda item: isinstance(item, type_), iterable)


def ceildiv(a, b, /):
    assert b != 0
    if b == 1:
        return a
    else:
        # See https://stackoverflow.com/a/17511341
        return -(a // -b)


def regexify(pattern: str):
    """Convert an expression pattern into a regex pattern.

    This is useful for testing.

    """
    # Escape common characters
    for char in ["(", ")", "[", "]", "*", "+"]:
        pattern = pattern.replace(char, f"\\{char}")

    # Convert '#' to '\d+' (to avoid numbering issues with arrays)
    pattern = pattern.replace("#", r"\d+")

    return pattern


def is_ellipsis_type(obj: Any) -> bool:
    return (
        obj is Ellipsis
        or (
            isinstance(obj, collections.abc.Sequence)
            and all(item is Ellipsis for item in obj)
        )
    )


@contextlib.contextmanager
def stack(list_, to_push):
    list_.extend(to_push)
    yield
    for _ in to_push:
        list_.pop(-1)


@contextlib.contextmanager
def dict_stack(dict_, to_push):
    for key, value in to_push.items():
        dict_[key] = value
    yield
    for key in to_push:
        dict_.pop(key)


def _get_method_cache(obj):
    if not hasattr(obj, "_pyop3_method_cache"):
        # Use object.__setattr__ to get around frozen dataclasses
        object.__setattr__(obj, "_pyop3_method_cache", collections.defaultdict(dict))
    return obj._pyop3_method_cache


def cached_method(*args, **kwargs):
    def wrapper(func):
        return cachetools.cachedmethod(
            lambda self: _get_method_cache(self)[func.__qualname__], *args, **kwargs
        )(func)
    return wrapper


def pretty_type(obj: Any) -> str:
    type_ = type(obj)
    return f"{type_.__module__}.{type_.__name__}"


@functools.singledispatch
def safe_equals(a, b, /) -> bool:
    return a == b


@safe_equals.register(np.ndarray)
def _(a, b, /) -> bool:
    return (a == b).all()
