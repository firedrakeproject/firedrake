# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""This module implements the infrastructure required for versioning
of data carrying objects (chiefly :class:`~pyop2.base.Dat`). Core
functionality provided includes object version numbers and copy on
write duplicates.

Each data carrying object is equipped with a version number. This is
incremented every time the value of the object is changed, whether
this is by a :func:`~pyop2.base.par_loop` or through direct user access to a
:attr:`data` attribute. Access to the :attr:`data_ro` read only data attribute does
not increase the version number.

Data carrying objects are also equipped with a :meth:`duplicate`
method. From a user perspective, this is a deep copy of the original
object. In the case of :class:`~pyop2.base.Dat` objects, this is implemented
as a shallow copy along with a copy on write mechanism which causes
the actual copy to occur if either the original or the copy is
modified.  The delayed copy is implemented by immediately creating a
copy :func:`~pyop2.base.par_loop` and, if lazy evaluation is enabled,
enqueing it. This ensures that the dependency trace will cause all
operations on which the copy depends to occur before the
copy. Conversely, the dependency of the copy :class:`~pyop2.base.Dat` on the
copying loop is artificially removed. This prevents the execution of
the copy being triggered when the copy :class:`~pyop2.base.Dat` is
read. Instead, writes to the original and copy :class:`~pyop2.base.Dat` are
intercepted and execution of the copy :func:`~pyop2.base.par_loop` is forced
at that point."""

from decorator import decorator
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


@decorator
def modifies(method, self, *args, **kwargs):
    "Decorator for methods that modify their instance's data"

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


@decorator
def modifies_arguments(func, *args, **kwargs):
    "Decorator for functions that modify their arguments' data"
    retval = func(*args, **kwargs)
    for a in args:
        if hasattr(a, 'access') and a.access != op2.READ:
            a.data._version_bump()
    return retval


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
