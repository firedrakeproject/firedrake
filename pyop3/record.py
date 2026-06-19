"""

Instantiating record types
--------------------------
* call record_setup()

Record IDs
----------
All records are given a unique ID (a string). This is useful as a cache key

In general if you call `record_init` on a record you will get an object with a
new ID. However, sometimes you want to preserve the ID (e.g. if you are only
reshaping something - the object is still considered to be the 'same'). To do
this you have to specify 'non_id_attrs'.

"""
from __future__ import annotations

import collections
import dataclasses
from collections.abc import Callable, Mapping
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
        cls.record_setup = _record_setup

        def _record_method_cache(self):
            return collections.defaultdict(dict)

        # if kwargs.get("frozen", False):
        #     cls.__hash__ = _frozenrecord_hash

        return cls
    return wrapper


def _record_init(self: Any, **attrs: Mapping[str,Any]) -> Any:
    new_attrs = {}
    attrs_changed = False
    change_id = False
    for field in dataclasses.fields(self):
        orig_attr = getattr(self, field.name)
        new_attr = attrs.pop(field.name, orig_attr)
        if not utils.safe_equals(new_attr, orig_attr):
            attrs_changed = True
            # We only have to change the ID for some attrs
            if field.name not in getattr(self, "non_id_attrs", set()):
                change_id = True
        new_attrs[field.name] = new_attr

    if attrs:
        valid_attr_names = tuple(field.name for field in dataclasses.fields(self))
        raise AssertionError(
            f"Unrecognised attributes: '{attrs.keys()}' are not in '{valid_attr_names}'"
        )

    if not attrs_changed:
        return self
    elif self.__dataclass_params__.frozen and not change_id:
        try:
            return _make_record_maybe_singleton(self, new_attrs, change_id=False)
        except UnhashableObjectException:
            return _make_record(self, new_attrs, change_id=False)
    else:
        return _make_record(self, new_attrs, change_id=change_id)


# NOTE: We use COMM_SELF because __record_init__ isn't always called collectively.
# I need to think harder about the legality of this. Should I disallow the comm attr
# for objects where this happens?
# @memory_cache(heavy=True, get_comm=lambda self, *a, **kw: self.comm or MPI.COMM_SELF)
# actually just disable this unless we can prove that it's necessary - it generates a
# lot of cache misses and probably slows up GC
# @memory_cache(heavy=True, get_comm=lambda *a, **kw: MPI.COMM_SELF)
def _make_record_maybe_singleton(*args, **kwargs):
    return _make_record(*args, **kwargs)


def _make_record(self, attrs, *, change_id: bool):
    new = object.__new__(type(self))
    for field_name, attr in attrs.items():
        object.__setattr__(new, field_name, attr)
    new_id = utils.unique_id(self) if change_id else self.record_id
    new.record_setup(_id=new_id)
    return new


def _record_setup(self, *, _id: str | None = None) -> None:
    """Finalise a record object.

    This method should be called inside every record ``__init__`` method.

    Parameters
    ----------
    _id
        The ID of the object. Internal use only.
    """
    if _id is None:
        _id = utils.unique_id(self)
    object.__setattr__(self, "record_id", _id)

    if hasattr(self, "__post_init__"):
        self.__post_init__()


def _frozenrecord_hash(self):
    if hasattr(self, "_cached_hash"):
        return self._cached_hash

    hash_ = hash(dataclasses.fields(self))
    object.__setattr__(self, "_cached_hash", hash_)
    return hash_


def attr(attr_name: str) -> property:
    return property(lambda self: getattr(self, attr_name))


