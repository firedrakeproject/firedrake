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

"""OP2 backend configuration and auxiliaries.

.. warning :: User code should usually set the backend via :func:`pyop2.op2.init`
"""

import void
import finalised
from logger import warning
from mpi import collective
backends = {'void': void, 'finalised': finalised}


def _make_object(obj, *args, **kwargs):
    """Instantiate `obj` with `*args` and `**kwargs`.
    This will instantiate an object of the correct type for the
    currently selected backend.  Use this over simple object
    instantiation if you want a generic superclass method to
    instantiate objects that at runtime should be of the correct
    backend type.

    As an example, let's say we want a method to zero a :class:`Dat`.
    This will look the same on all backends::

      def zero(self):
          ParLoop(self._zero_kernel, self.dataset.set,
                  self(WRITE)).compute()

    but if we place this in a base class, then the :class:`ParLoop`
    object we instantiate is a base `ParLoop`, rather than (if we're
    on the sequential backend) a sequential `ParLoop`.  Instead, you
    should do this::

      def zero(self):
          _make_object('ParLoop', self._zero_kernel, self.dataset.set,
                       self(WRITE)).compute()

    That way, the correct type of `ParLoop` will be instantiated at
    runtime."""
    return _BackendSelector._backend.__dict__[obj](*args, **kwargs)


class _BackendSelector(type):

    """Metaclass creating the backend class corresponding to the requested
    class."""

    _backend = void

    def __new__(cls, name, bases, dct):
        """Inherit Docstrings when creating a class definition. A variation of
        http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
        by Paul McGuire
        Source: http://stackoverflow.com/a/8101118/396967
        """

        # Get the class docstring
        if not('__doc__' in dct and dct['__doc__']):
            for mro_cls in (cls for base in bases for cls in base.mro()):
                doc = mro_cls.__doc__
                if doc:
                    dct['__doc__'] = doc
                    break
        # Get the attribute docstrings
        for attr, attribute in dct.items():
            if not attribute.__doc__:
                for mro_cls in (cls for base in bases for cls in base.mro()
                                if hasattr(cls, attr)):
                    doc = getattr(getattr(mro_cls, attr), '__doc__')
                    if doc:
                        attribute.__doc__ = doc
                        break
        return type.__new__(cls, name, bases, dct)

    def __call__(cls, *args, **kwargs):
        """Create an instance of the request class for the current backend"""

        # Try the selected backend first
        try:
            t = cls._backend.__dict__[cls.__name__]
        except KeyError as e:
            warning('Backend %s does not appear to implement class %s'
                    % (cls._backend.__name__, cls.__name__))
            raise e
        # Invoke the constructor with the arguments given
        return t(*args, **kwargs)

    # More disgusting metaclass voodoo
    def __instancecheck__(cls, instance):
        """Return True if instance is an instance of cls

        We need to override the default isinstance check because
        `type(op2.Set(10))` is `base.Set` but type(op2.Set) is
        `_BackendSelector` and so by default `isinstance(op2.Set(10),
        op2.Set)` is False.

        """
        return isinstance(instance, cls._backend.__dict__[cls.__name__])

    def __subclasscheck__(cls, subclass):
        """Return True if subclass is a subclass of cls

        We need to override the default subclass check because
        type(op2.Set(10)) is `base.Set` but type(op2.Set) is
        `_BackendSelector` and so by default
        `isinstance(type(op2.Set(10)), op2.Set)` is False.

        """
        return issubclass(subclass, cls._backend.__dict__[cls.__name__])

    def fromhdf5(cls, *args, **kwargs):
        try:
            return cls._backend.__dict__[cls.__name__].fromhdf5(*args, **kwargs)
        except AttributeError as e:
            warning("op2 object %s does not implement fromhdf5 method" % cls.__name__)
            raise e


def get_backend():
    """Get the OP2 backend"""

    return _BackendSelector._backend.__name__


@collective
def set_backend(backend):
    """Set the OP2 backend"""

    global _BackendSelector
    if _BackendSelector._backend != void:
        raise RuntimeError("The backend can only be set once!")

    mod = backends.get(backend)
    if mod is None:
        try:
            # We need to pass a non-empty fromlist so that __import__
            # returns the submodule (i.e. the backend) rather than the
            # package.
            mod = __import__('pyop2.%s' % backend, fromlist=[None])
        except ImportError as e:
            warning('Unable to import backend %s' % backend)
            raise e
    backends[backend] = mod
    _BackendSelector._backend = mod


@collective
def unset_backend():
    """Unset the OP2 backend"""
    _BackendSelector._backend = finalised
