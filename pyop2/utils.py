# This file is part of PyOP2.
#
# PyOP2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PyOP2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PyOP2.  If not, see <http://www.gnu.org/licenses>
#
# Copyright (c) 2011, Graham Markall <grm08@doc.ic.ac.uk> and others. Please see
# the AUTHORS file in the main source directory for a full list of copyright
# holders.

"""Common utility classes/functions."""

import numpy as np

from exceptions import DataTypeError, DataValueError

def as_tuple(item, type=None, length=None):
    # Empty list if we get passed None
    if item is None:
        t = []
    else:
        # Convert iterable to list...
        try:
            t = tuple(item)
        # ... or create a list of a single item
        except TypeError:
            t = (item,)*(length or 1)
    if length and not len(t) == length:
        raise ValueError("Tuple needs to be of length %d" % length)
    if type and not all(isinstance(i, type) for i in t):
        raise TypeError("Items need to be of type %s" % type)
    return t

class validate_base:
    """Decorator to validate arguments

    Formal parameters that don't exist in the definition of the function
    being decorated as well as actual arguments not being present when
    the validation is called are silently ignored."""

    def __init__(self, *checks):
        self._checks = checks

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            self.varnames = f.func_code.co_varnames
            self.file = f.func_code.co_filename
            self.line = f.func_code.co_firstlineno+1
            self.check_args(args, kwargs)
            return f(*args, **kwargs)
        return wrapper

    def check_args(self, args, kwargs):
        for argname, argcond, exception in self._checks:
            # If the argument argname is not present in the decorated function
            # silently ignore it
            try:
                i = self.varnames.index(argname)
            except ValueError:
                # No formal parameter argname
                continue
            # Try the argument by keyword first, and by position second.
            # If the argument isn't given, silently ignore it.
            try:
                arg = kwargs.get(argname)
                arg = arg or args[i]
            except IndexError:
                # No actual parameter argname
                continue
            self.check_arg(arg, argcond, exception)

class validate_type(validate_base):
    """Decorator to validate argument types

    The decorator expects one or more arguments, which are 3-tuples of
    (name, type, exception), where name is the argument name in the
    function being decorated, type is the argument type to be validated
    and exception is the exception type to be raised if validation fails."""

    def check_arg(self, arg, argtype, exception):
        if not isinstance(arg, argtype):
            raise exception("%s:%d Parameter %s must be of type %r" \
                    % (self.file, self.line, arg, argtype))

def verify_reshape(data, dtype, shape, allow_none=False):
    """Verify data is of type dtype and try to reshaped to shape."""

    if data is None and allow_none:
        try:
            return np.asarray([], dtype=np.dtype(dtype))
        except TypeError:
            raise DataTypeError("Invalid data type: %s" % dtype)
    elif data is None:
        raise DataValueError("Invalid data: None is not allowed!")
    else:
        t = np.dtype(dtype) if dtype is not None else None
        try:
            a = np.asarray(data, dtype=t)
        except ValueError:
            raise DataValueError("Invalid data: cannot convert to %s!" % dtype)
        try:
            return a.reshape(shape)
        except ValueError:
            raise DataValueError("Invalid data: expected %d values, got %d!" % \
                    (np.prod(shape), np.asarray(data).size))
