from __future__ import annotations

import collections
import dataclasses
from collections.abc import Mapping
from typing import Any

from mpi4py import MPI

from pyop3 import utils
from pyop3.cache import memory_cache
from pyop3.exceptions import UnhashableObjectException


def record():
    return _make_record_class(eq=False)


def frozenrecord():
    return _make_record_class(frozen=True)


def _make_record_class(**kwargs):
    def wrapper(cls):
        cls = dataclasses.dataclass(**kwargs)(cls)
        cls.__record_init__ = _record_init

        def _record_method_cache(self):
            return collections.defaultdict(dict)

        # if kwargs.get("frozen", False):
        #     cls.__hash__ = _frozenrecord_hash

        return cls
    return wrapper


def _record_init(self: Any, **attrs: Mapping[str,Any]) -> Any:
    new_attrs = {}
    attrs_changed = False
    for field in dataclasses.fields(self):
        orig_attr = getattr(self, field.name)
        new_attr = attrs.pop(field.name, orig_attr)
        if not utils.safe_equals(new_attr, orig_attr):
            attrs_changed = True
        new_attrs[field.name] = new_attr

    if attrs:
        valid_attr_names = tuple(field.name for field in dataclasses.fields(self))
        raise AssertionError(
            f"Unrecognised attributes: '{attrs.keys()}' are not in '{valid_attr_names}'"
        )

    if not attrs_changed:
        return self
    elif self.__dataclass_params__.frozen:
        try:
            return _make_record_maybe_singleton(self, new_attrs)
        except UnhashableObjectException:
            return _make_record(self, new_attrs)
    else:
        return _make_record(self, new_attrs)


# NOTE: We use COMM_SELF because __record_init__ isn't always called collectively.
# I need to think harder about the legality of this. Should I disallow the comm attr
# for objects where this happens?
# @memory_cache(heavy=True, get_comm=lambda self, *a, **kw: self.comm or MPI.COMM_SELF)
@memory_cache(heavy=True, get_comm=lambda *a, **kw: MPI.COMM_SELF)
def _make_record_maybe_singleton(*args, **kwargs):
    return _make_record(*args, **kwargs)


def _make_record(self, attrs):
    new = object.__new__(type(self))
    for field_name, attr in attrs.items():
        object.__setattr__(new, field_name, attr)

    if hasattr(new, "__post_init__"):
        new.__post_init__()

    return new


def _frozenrecord_hash(self):
    if hasattr(self, "_cached_hash"):
        return self._cached_hash

    hash_ = hash(dataclasses.fields(self))
    object.__setattr__(self, "_cached_hash", hash_)
    return hash_


def attr(attr_name: str) -> property:
    return property(lambda self: getattr(self, attr_name))


