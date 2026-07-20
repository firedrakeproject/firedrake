from __future__ import annotations

import collections
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from mpi4py import MPI

from pyop3 import utils
from pyop3.cache import memory_cache
from pyop3.exceptions import UnhashableObjectException


def record(**kwargs):
    assert "eq" not in kwargs
    assert "repr" not in kwargs
    return _make_record_class(eq=False, repr=False, **kwargs)


def frozenrecord(**kwargs):
    assert "frozen" not in kwargs
    return _make_record_class(frozen=True, **kwargs)


def _make_record_class(**kwargs):
    def wrapper(cls):
        cls = dataclasses.dataclass(**kwargs)(cls)

        cls.record_new = _record_new
        cls.record_init = _record_init

        old_init = cls.__init__

        def custom_init(self, *args, **kwargs):
            # print("calling old init")
            old_init(self, *args, **kwargs)
            # print("called")

        cls.__init__ = custom_init


        # def _record_method_cache(self):
        #     return collections.defaultdict(dict)

        # if kwargs.get("frozen", False):
        #     cls.__hash__ = _frozenrecord_hash

        return cls
    return wrapper


@classmethod
def _record_new(cls, **attrs: Mapping[str,Any]) -> Any:
    """Create and initialise a new record."""
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

    new = object.__new__(type(self))

    if self.__dataclass_params__.frozen and not change_id:
        try:
            record = _make_record_maybe_singleton(self, new_attrs, change_id=False)
        except UnhashableObjectException:
            record = _make_record(self, new_attrs, change_id=False)
    else:
        record = _make_record(self, new_attrs, change_id=change_id)

    if hasattr(record, "__post_init__"):
        record.__post_init__()

    return record


# NOTE: We use COMM_SELF because __record_init__ isn't always called collectively.
# I need to think harder about the legality of this. Should I disallow the comm attr
# for objects where this happens?
# @memory_cache(heavy=True, get_comm=lambda self, *a, **kw: self.comm or MPI.COMM_SELF)
# actually just disable this unless we can prove that it's necessary - it generates a
# lot of cache misses and probably slows up GC
# @memory_cache(heavy=True, get_comm=lambda self, *a, **kw: self._comm or MPI.COMM_SELF)
# def _make_record_maybe_singleton(self, *args, **kwargs):
#     return _make_record(self, *args, **kwargs)


def _record_init(self, attrs: Mapping[str, Any]) -> None:
    """Initialise a new record."""
    for field_name, attr in attrs.items():
        object.__setattr__(self, field_name, attr)


# def _frozenrecord_hash(self):
#     if hasattr(self, "_cached_hash"):
#         return self._cached_hash
#
#     hash_ = hash(dataclasses.fields(self))
#     object.__setattr__(self, "_cached_hash", hash_)
#     return hash_


def attr(attr_name: str) -> property:
    return property(lambda self: getattr(self, attr_name))
