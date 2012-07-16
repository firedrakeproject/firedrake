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

"""OP2 backend configuration and auxiliaries."""

backends = {}
try:
    import cuda
    backends['cuda'] = cuda
except ImportError, e:
    from warnings import warn
    warn("Unable to import cuda backend: %s" % str(e))

try:
    import opencl
    backends['opencl'] = opencl
except ImportError, e:
    from warnings import warn
    warn("Unable to import opencl backend: %s" % str(e))

import sequential
import void

backends['sequential'] = sequential
backends['void'] = void

class BackendSelector(type):
    """Metaclass creating the backend class corresponding to the requested
    class."""

    _backend = void
    _defaultbackend = sequential

    def __new__(cls, name, bases, dct):
        """Inherit Docstrings when creating a class definition. A variation of
        http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
        by Paul McGuire
        Source: http://stackoverflow.com/a/8101118/396967
        """

        # Get the class docstring
        if not('__doc__' in dct and dct['__doc__']):
            for mro_cls in (cls for base in bases for cls in base.mro()):
                doc=mro_cls.__doc__
                if doc:
                    dct['__doc__']=doc
                    break
        # Get the attribute docstrings
        for attr, attribute in dct.items():
            if not attribute.__doc__:
                for mro_cls in (cls for base in bases for cls in base.mro()
                                if hasattr(cls, attr)):
                    doc=getattr(getattr(mro_cls,attr),'__doc__')
                    if doc:
                        attribute.__doc__=doc
                        break
        return type.__new__(cls, name, bases, dct)

    def __call__(cls, *args, **kwargs):
        """Create an instance of the request class for the current backend"""

        # Try the selected backend first
        try:
            t = cls._backend.__dict__[cls.__name__]
        # Fall back to the default (i.e. sequential) backend
        except KeyError:
            t = cls._defaultbackend.__dict__[cls.__name__]
        # Invoke the constructor with the arguments given
        return t(*args, **kwargs)

def get_backend():
    """Get the OP2 backend"""

    return BackendSelector._backend.__name__

def set_backend(backend):
    """Set the OP2 backend"""

    global BackendSelector
    if BackendSelector._backend != void:
        raise RuntimeError("The backend can only be set once!")
    if backend not in backends:
        raise ValueError("backend must be one of %r" % backends.keys())
    BackendSelector._backend = backends[backend]

def par_loop(kernel, it_space, *args):
    return BackendSelector._backend.par_loop(kernel, it_space, *args)
