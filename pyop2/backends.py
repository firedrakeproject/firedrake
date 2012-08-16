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

class _BackendSelector(type):
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

    return _BackendSelector._backend.__name__

def set_backend(backend):
    """Set the OP2 backend"""

    global _BackendSelector
    if _BackendSelector._backend != void:
        raise RuntimeError("The backend can only be set once!")
    if backend not in backends:
        raise ValueError("backend must be one of %r" % backends.keys())
    _BackendSelector._backend = backends[backend]

def unset_backend():
    """Unset the OP2 backend"""
    _BackendSelector._backend = void

def par_loop(kernel, it_space, *args):
    return _BackendSelector._backend.par_loop(kernel, it_space, *args)

def solve(M, x, b):
    return _BackendSelector._backend.solve(M, x, b)
