from __future__ import annotations

import dataclasses
from typing import Any

import pyop3.cache
import pyop3.obj
from pyop3 import utils


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

        # The __new__ to __init__ pipeline is quite confusing. By default, if
        # __new__ returns an object with the same type as the original class,
        # then __init__ is called on the resulting object using the *same
        # arguments that were passed to __new__*. To demonstrate why this is
        # an issue, consider the following case using pyop3.Mul nodes:
        #
        #     Mul(Mul("a", "b"), 1)
        #
        # We would like it such that the outer Mul (inside __new__) recognises
        # that multiplying by 1 is identity and to then return a bare
        # Mul("a", "b"). However, *this is the same type as the original caller*.
        # This means that Mul.__init__ will then be called on an already
        # initialised object, which is wrong.
        #
        # To get around this we require that, if an object defines __new__,
        # then we disallow it to have an __init__ method. It is therefore the
        # responsibility of __new__ to return a fully initialised object.
        #
        # As a final complication, it now becomes possible to call cls.__new__
        # from both a 'regular' constructor call, and from record_new. These may
        # have different interfaces and it is the responsibility of the class
        # writer to makes things work.
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

        # Make sure that we call all the finalisers after initialisation
        if cls.__init__ is object.__init__:
            def init_with_finalizers(self, *args, **kwargs):
                object.__init__(self)
                _finalize_obj(self)
        else:
            orig_init = cls.__init__
            def init_with_finalizers(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                _finalize_obj(self)
        cls.__init__ = init_with_finalizers

        # Attach the other init method
        cls.record_new = _record_new

        # Cache the hash for frozen objects
        if cls.__dataclass_params__.frozen:
            orig_hash = cls.__hash__
            def frozen_hash(self):
                if not hasattr(self, "_record_cached_hash"):
                    object.__setattr__(self, "_record_cached_hash", orig_hash(self))
                return self._record_cached_hash
            cls.__hash__ = frozen_hash

        return cls

    return wrapper


def _record_new(self, **attrs: Any) -> Any:
    """Create and initialise a new record from an existing one."""
    new_attrs = {}
    attrs_changed = False
    for field in dataclasses.fields(self):
        orig_attr = getattr(self, field.name)
        if field.name in attrs:
            new_attr = attrs.pop(field.name, orig_attr)
            if not utils.safe_equals(new_attr, orig_attr):
                attrs_changed = True
        else:
            new_attr = orig_attr
        new_attrs[field.name] = new_attr

    if attrs:
        valid_attr_names = tuple(field.name for field in dataclasses.fields(self))
        raise AssertionError(
            f"Unrecognised attributes: '{attrs.keys()}' are not in '{valid_attr_names}'"
        )

    # If we haven't changed anything and the object is frozen then just hand it back
    if not attrs_changed and self.__dataclass_params__.frozen:
        return self

    cls = type(self)
    new = cls.__new__(cls, **new_attrs)

    # If a class defines a custom __new__ method then we assume all
    # initialisation is done there
    if cls.__new__ is object.__new__:
        for attr_name, new_attr in new_attrs.items():
            object.__setattr__(new, attr_name, new_attr)

    _finalize_obj(new)
    return new


def _finalize_obj(obj: pyop3.obj.Object) -> None:
    # Call any finaliser functions. Because inheritance is complicated we ignore
    # it and call all of the finalisers we find in the MRO.
    for type_ in type(obj).__mro__:
        if hasattr(type_, _demangle(type_, "__record_post_init")):
            getattr(type_, _demangle(type_, "__record_post_init"))(obj)


# Now we have abstract attrs I don't think we need this any more
def attr(attr_name: str) -> property:
    return property(lambda self: getattr(self, attr_name))


def _get_abstract_attrs_per_class(cls: type) -> tuple:
    # Undo the name mangling that the double underscore introduces
    return getattr(cls, _demangle(cls, "__abstract_record_attrs"), ())


def _get_abstract_attrs(cls: type) -> tuple:
    assert not _get_abstract_attrs_per_class(cls), \
        "Final class should not define any abstract attributes"
    attrs = []
    for parent_class in cls.__mro__[1:]:
        attrs.extend(_get_abstract_attrs_per_class(parent_class))
    return tuple(attrs)


def _demangle(cls: type, attr: str) -> str:
    """Undo Python name mangling."""
    assert attr.startswith("__")
    return f"_{cls.__name__}{attr}"
