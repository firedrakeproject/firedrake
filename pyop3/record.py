from __future__ import annotations

import collections
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from mpi4py import MPI

import pyop3.cache
import pyop3.obj


def record(**kwargs):
    assert "eq" not in kwargs
    assert "repr" not in kwargs
    return _make_record_class(maybe_singleton=False, eq=False, repr=False, **kwargs)


def frozenrecord(maybe_singleton: bool = True, **kwargs):
    assert "frozen" not in kwargs
    return _make_record_class(maybe_singleton=maybe_singleton, frozen=True, **kwargs)


def _make_record_class(*, maybe_singleton: bool, **kwargs):

    def wrapper(cls):
        assert issubclass(cls, pyop3.obj.Object)

        assert "init" not in kwargs
        cls = dataclasses.dataclass(init=False, **kwargs)(cls)

        if maybe_singleton:
            assert kwargs.get("frozen", False)
            cls._record_maybe_singleton = True
        else:
            cls._record_maybe_singleton = False

        assert cls.__init__ is object.__init__, \
            f"'{cls.__qualname__}' should not define its own '__init__'"

        # Overload __new__ unless the class already does so (and it
        # must then call '_create_record' or '_maybe_create_frozenrecord')
        if cls.__new__ is object.__new__:
            cls.__new__ = _record_dunder_new_maybe_pickle

        # attach other methods
        if not hasattr(cls, "record_prepare_args"):
            cls.record_prepare_args = _record_prepare_args_default
        cls.record_new = _record_new

        # if kwargs.get("frozen", False):
        #     cls.__hash__ = _frozenrecord_hash

        return cls

    return wrapper


def _record_dunder_new_maybe_pickle(cls, *args, **kwargs):
    # if not args and not kwargs:
    if False:
        return object.__new__(cls)
    else:
        return _record_dunder_new(cls, *args, **kwargs)


def _record_dunder_new(cls, *args, _record_args_prepared: bool = False, **kwargs) -> Any:
    if not _record_args_prepared:
        kwargs = cls.record_prepare_args(*args, **kwargs)
    else:
        assert not args
    if cls.__dataclass_params__.frozen and cls._record_maybe_singleton:
        return _maybe_create_frozenrecord(cls, **kwargs)
    else:
        return _create_record(cls, **kwargs)


@pyop3.cache.memory_cache(
    heavy=True,
    get_comm=lambda cls, **attrs: cls.get_comm(**attrs),
)
def _maybe_create_frozenrecord(cls: Any, **attrs: Any) -> Any:
    return _create_record(cls, **attrs)


def _create_record(cls: Any, **attrs: Any) -> Any:
    self = object.__new__(cls)
    for field_name, attr in attrs.items():
        object.__setattr__(self, field_name, attr)
    if hasattr(self, "__post_init__"):
        self.__post_init__()
    return self


@classmethod
def _record_prepare_args_default(cls, *args, **kwargs):
    assert len(args) <= len(dataclasses.fields(cls))

    attrs = {}
    # consume all args
    for arg, field in zip(args, dataclasses.fields(cls)):
        # TODO: assert no kw_only etc
        attrs[field.name] = arg
    return attrs | kwargs


def _record_new(self, **attrs: Any) -> Any:
    """Create and initialise a new record from an existing one."""
    new_attrs = {}
    for field in dataclasses.fields(self):
        orig_attr = getattr(self, field.name)
        new_attr = attrs.pop(field.name, orig_attr)
        new_attrs[field.name] = new_attr

    if attrs:
        valid_attr_names = tuple(field.name for field in dataclasses.fields(self))
        raise AssertionError(
            f"Unrecognised attributes: '{attrs.keys()}' are not in '{valid_attr_names}'"
        )

    return type(self)(_record_args_prepared=True, **new_attrs)


# def _frozenrecord_hash(self):
#     if hasattr(self, "_cached_hash"):
#         return self._cached_hash
#
#     hash_ = hash(dataclasses.fields(self))
#     object.__setattr__(self, "_cached_hash", hash_)
#     return hash_


def attr(attr_name: str) -> property:
    return property(lambda self: getattr(self, attr_name))
