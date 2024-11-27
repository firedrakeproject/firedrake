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

"""Common utility classes/functions."""


import os
import sys
import numpy as np
from decorator import decorator
import argparse
import petsc4py

from functools import cached_property  # noqa: F401

from pyop2.exceptions import DataTypeError, DataValueError
from pyop2.configuration import configuration


def as_tuple(item, type=None, length=None, allow_none=False):
    # Empty list if we get passed None
    if item is None:
        t = ()
    else:
        # Convert iterable to tuple...
        try:
            t = tuple(item)
        # ... or create a list of a single item
        except (TypeError, NotImplementedError):
            t = (item,) * (length or 1)
    if configuration["type_check"]:
        if length and not len(t) == length:
            raise ValueError("Tuple needs to be of length %d" % length)
        if type is not None:
            if allow_none:
                valid = all((isinstance(i, type) or i is None) for i in t)
            else:
                valid = all(isinstance(i, type) for i in t)
            if not valid:
                raise TypeError("Items need to be of type %s" % type)
    return t


def as_type(obj, typ):
    """Return obj if it is of dtype typ, otherwise return a copy type-cast to
    typ."""
    # Assume it's a NumPy data type
    try:
        return obj if obj.dtype == typ else obj.astype(typ)
    except AttributeError:
        if isinstance(obj, int):
            return np.int64(obj).astype(typ)
        elif isinstance(obj, float):
            return np.float64(obj).astype(typ)
        else:
            raise TypeError("Invalid type %s" % type(obj))


def tuplify(xs):
    """Turn a data structure into a tuple tree."""
    try:
        return tuple(tuplify(x) for x in xs)
    except TypeError:
        return xs


class validate_base:

    """Decorator to validate arguments

    Formal parameters that don't exist in the definition of the function
    being decorated as well as actual arguments not being present when
    the validation is called are silently ignored."""

    def __init__(self, *checks):
        self._checks = checks

    def __call__(self, f):
        def wrapper(f, *args, **kwargs):
            if configuration["type_check"]:
                self.nargs = f.__code__.co_argcount
                self.defaults = f.__defaults__ or ()
                self.varnames = f.__code__.co_varnames
                self.file = f.__code__.co_filename
                self.line = f.__code__.co_firstlineno + 1
                self.check_args(args, kwargs)
            return f(*args, **kwargs)
        return decorator(wrapper, f)

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
            # If the argument has a default value, also accept that (since the
            # constructor will be able to deal with that)
            default_index = i - self.nargs + len(self.defaults)
            if default_index >= 0 and arg == self.defaults[default_index]:
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
            raise exception("%s:%d Parameter %s must be of type %r"
                            % (self.file, self.line, arg, argtype))


class validate_in(validate_base):

    """Decorator to validate argument is in a set of valid argument values

    The decorator expects one or more arguments, which are 3-tuples of
    (name, list, exception), where name is the argument name in the
    function being decorated, list is the list of valid argument values
    and exception is the exception type to be raised if validation fails."""

    def check_arg(self, arg, values, exception):
        if arg not in values:
            raise exception("%s:%d %s must be one of %s"
                            % (self.file, self.line, arg, values))


class validate_range(validate_base):

    """Decorator to validate argument value is in a given numeric range

    The decorator expects one or more arguments, which are 3-tuples of
    (name, range, exception), where name is the argument name in the
    function being decorated, range is a 2-tuple defining the valid argument
    range and exception is the exception type to be raised if validation
    fails."""

    def check_arg(self, arg, range, exception):
        if not range[0] <= arg <= range[1]:
            raise exception("%s:%d %s must be within range %s"
                            % (self.file, self.line, arg, range))


class validate_dtype(validate_base):

    """Decorator to validate argument value is in a valid Numpy dtype

    The decorator expects one or more arguments, which are 3-tuples of
    (name, _, exception), where name is the argument name in the
    function being decorated, second argument is ignored and exception
    is the exception type to be raised if validation fails."""

    def check_arg(self, arg, ignored, exception):
        try:
            np.dtype(arg)
        except TypeError:
            raise exception("%s:%d %s must be a valid dtype"
                            % (self.file, self.line, arg))


def verify_reshape(data, dtype, shape, allow_none=False):
    """Verify data is of type dtype and try to reshaped to shape."""

    try:
        t = np.dtype(dtype) if dtype is not None else None
    except TypeError:
        raise DataTypeError("Invalid data type: %s" % dtype)
    if data is None and allow_none:
        return np.asarray([], dtype=t)
    elif data is None:
        raise DataValueError("Invalid data: None is not allowed!")
    else:
        try:
            a = np.asarray(data, dtype=t)
        except ValueError:
            raise DataValueError("Invalid data: cannot convert to %s!" % dtype)
        except TypeError:
            raise DataTypeError("Invalid data type: %s" % dtype)
        try:
            # Destructively modify shape.  Fails if data are not
            # contiguous, but that's what we want anyway.
            a.shape = shape
            return a
        except ValueError:
            raise DataValueError("Invalid data: expected %d values, got %d!" %
                                 (np.prod(shape), np.asarray(data).size))


def align(bytes, alignment=16):
    """Align BYTES to a multiple of ALIGNMENT"""
    return ((bytes + alignment - 1) // alignment) * alignment


def flatten(iterable):
    """Flatten a given nested iterable."""
    return (x for e in iterable for x in e)


def parser(description=None, group=False):
    """Create default argparse.ArgumentParser parser for pyop2 programs."""
    parser = argparse.ArgumentParser(description=description,
                                     add_help=True,
                                     prefix_chars="-",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    g = parser.add_argument_group(
        'pyop2', 'backend configuration options') if group else parser

    g.add_argument('-d', '--debug', default=argparse.SUPPRESS,
                   type=int, choices=list(range(8)),
                   help='set debug level' if group else 'set pyop2 debug level')
    g.add_argument('-l', '--log-level', default='WARN',
                   choices=['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'],
                   help='set logging level (default=WARN)' if group else
                   'set pyop2 logging level (default=WARN)')

    return parser


def parse_args(*args, **kwargs):
    """Return parsed arguments as variables for later use.

    ARGS and KWARGS are passed into the parser instantiation.
    The only recognised options are `group` and `description`."""
    return vars(parser(*args, **kwargs).parse_args())


def trim(docstring):
    """Trim a docstring according to `PEP 257
    <http://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation>`_."""
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)


def strip(code):
    return '\n'.join([l for l in code.splitlines() if l.strip() and l.strip() != ';'])


def get_petsc_dir():
    """Attempts to find the PETSc directory on the system
    """
    petsc_config = petsc4py.get_config()
    petsc_dir = petsc_config["PETSC_DIR"]
    petsc_arch = petsc_config["PETSC_ARCH"]
    pathlist = [petsc_dir]
    if petsc_arch:
        pathlist.append(os.path.join(petsc_dir, petsc_arch))
    return tuple(pathlist)


def get_petsc_variables():
    """Attempts obtain a dictionary of PETSc configuration settings
    """
    path = [get_petsc_dir()[-1], "lib/petsc/conf/petscvariables"]
    variables_path = os.path.join(*path)
    with open(variables_path) as fh:
        # Split lines on first '=' (assignment)
        splitlines = (line.split("=", maxsplit=1) for line in fh.readlines())
    return {k.strip(): v.strip() for k, v in splitlines}
