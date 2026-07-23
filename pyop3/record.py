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
    return _make_record_class(eq=False, repr=False, **kwargs)


def frozenrecord(**kwargs):
    assert "frozen" not in kwargs
    return _make_record_class(frozen=True, **kwargs)


def _make_record_class(**kwargs):

    def wrapper(cls):
        assert issubclass(cls, pyop3.obj.Object)

        # If we have a custom '__new__' then we cannot also have an
        # '__init__' method
        assert "init" not in kwargs
        if cls.__new__ is not object.__new__:
            assert cls.__init__ is object.__init__
            init = False
        else:
            init = True
        cls = dataclasses.dataclass(init=init, **kwargs)(cls)

        # Check that abstract attributes are implemented
        for abstract_attr in _get_abstract_attrs(cls):
            assert abstract_attr in cls.__dataclass_fields__, \
                f"class '{cls.__qualname__}' does not have attribute '{abstract_attr}'"

        if hasattr(cls, "record_prepare_args"):
            print("no longer recommended")

            # default
            initialinit = cls.__init__

            def old_init(self, *args, **kwargs):
                attrs = self.record_prepare_args(*args, **kwargs)
                initialinit(self, **attrs)

            cls.__init__ = old_init

        cls.record_new = _record_new

        # if kwargs.get("frozen", False):
        #     cls.__hash__ = _frozenrecord_hash

        return cls

    return wrapper


def _create_record(cls: Any, **attrs: Any) -> Any:
    self = object.__new__(cls)
    for field_name, attr in attrs.items():
        object.__setattr__(self, field_name, attr)

    # Run all __post_init__ methods
    # TODO: make record_post_init?
    for type_ in cls.__mro__:
        if hasattr(type_, "__post_init__"):
            type_.__post_init__(self)
    return self


def _record_new(self, **attrs: Any) -> Any:
    """Create and initialise a new record from an existing one."""
    cls = type(self)
    new = cls.__new__(cls, **attrs)

    # If a class defines a custom __new__ method then we assume all
    # initialisation is done there
    if cls.__new__ is object.__new__:
        for field in dataclasses.fields(self):
            orig_attr = getattr(self, field.name)
            new_attr = attrs.pop(field.name, orig_attr)
            object.__setattr__(new, field.name, new_attr)

        if attrs:
            valid_attr_names = tuple(field.name for field in dataclasses.fields(self))
            raise AssertionError(
                f"Unrecognised attributes: '{attrs.keys()}' are not in '{valid_attr_names}'"
            )

    return new



# def _frozenrecord_hash(self):
#     if hasattr(self, "_cached_hash"):
#         return self._cached_hash
#
#     hash_ = hash(dataclasses.fields(self))
#     object.__setattr__(self, "_cached_hash", hash_)
#     return hash_


# Now we have abstract attrs I don't think we need this any more
def attr(attr_name: str) -> property:
    return property(lambda self: getattr(self, attr_name))


def _get_abstract_attrs_per_class(cls: type) -> tuple:
    # Undo the name mangling that the double underscore introduces
    return getattr(cls, f"_{cls.__name__}__abstract_record_attrs", ())


def _get_abstract_attrs(cls: type) -> tuple:
    assert not _get_abstract_attrs_per_class(cls), \
        "Final class should not define any abstract attributes"
    attrs = []
    for parent_class in cls.__mro__[1:]:
        attrs.extend(_get_abstract_attrs_per_class(parent_class))
    return tuple(attrs)
