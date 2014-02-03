from copy import copy as shallow_copy
import op2


class Versioned(object):
    """Versioning class for objects with mutable data"""

    def __new__(cls, *args, **kwargs):
        obj = super(Versioned, cls).__new__(cls)
        obj._version = 1
        obj._version_before_zero = 1
        return obj

    def _version_bump(self):
        """Increase the data._version associated with this object. It should
        rarely, if ever, be necessary for a user to call this manually."""

        self._version_before_zero += 1
        # Undo_version = 0
        self._version = self._version_before_zero

    def _version_set_zero(self):
        """Set the data version of this object to zero (usually when
        self.zero() is called)."""
        self._version = 0


def modifies(method):
    "Decorator for methods that modify their instance's data"
    def inner(self, *args, **kwargs):
        # self is likely going to change

        # If I am a copy-on-write duplicate, I need to become real
        if hasattr(self, '_cow_is_copy_of') and self._cow_is_copy_of:
            original = self._cow_is_copy_of
            self._cow_actual_copy(original)
            self._cow_is_copy_of = None
            original._cow_copies.remove(self)

        # If there are copies of me, they need to become real now
        if hasattr(self, '_cow_copies'):
            for c in self._cow_copies:
                c._cow_actual_copy(self)
                c._cow_is_copy_of = None
            self._cow_copies = []

        retval = method(self, *args, **kwargs)

        self._version_bump()

        return retval

    return inner


def modifies_arguments(func):
    "Decorator for functions that modify their arguments' data"
    def inner(*args, **kwargs):
        retval = func(*args, **kwargs)
        for a in args:
            if hasattr(a, 'access') and a.access != op2.READ:
                a.data._version_bump()
        return retval
    return inner


class CopyOnWrite(object):
    """
    Class that overrides the duplicate method and performs the actual copy
    operation when either the original or the copy has been written.  Classes
    that inherit from CopyOnWrite need to provide the methods:

    _cow_actual_copy(self, src):
        Performs an actual copy of src's data to self

    _cow_shallow_copy(self):
        Returns a shallow copy of the current object, e.g. the data handle
        should be the same.
        (optionally, otherwise the standard copy.copy() is used)
    """

    def duplicate(self):
        if hasattr(self, '_cow_shallow_copy'):
            dup = self._cow_shallow_copy()
        else:
            dup = shallow_copy(self)

        if not hasattr(self, '_cow_copies'):
            self._cow_copies = []
        self._cow_copies.append(dup)
        dup._cow_is_copy_of = self

        return dup
