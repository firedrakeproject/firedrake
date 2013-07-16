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

"""Provides common base classes for cached objects."""

import cPickle
import gzip
import os
from ir.ast_base import Node
from copy import copy as shallow_copy
import op2
from logger import debug
from ufl.algorithms.signature import compute_form_signature


class Cached(object):

    """Base class providing global caching of objects. Derived classes need to
    implement classmethods :meth:`_process_args` and :meth:`_cache_key`
    and define a class attribute :attr:`_cache` of type :class:`dict`.

    .. warning::
        The derived class' :meth:`__init__` is still called if the object is
        retrieved from cache. If that is not desired, derived classes can set
        a flag indicating whether the constructor has already been called and
        immediately return from :meth:`__init__` if the flag is set. Otherwise
        the object will be re-initialized even if it was returned from cache!
    """

    def __new__(cls, *args, **kwargs):
        args, kwargs = cls._process_args(*args, **kwargs)
        key = cls._cache_key(*args, **kwargs)
        try:
            return cls._cache_lookup(key)
        except KeyError:
            obj = super(Cached, cls).__new__(cls)
            obj._key = key
            obj._initialized = False
            # obj.__init__ will be called twice when constructing
            # something not in the cache.  The first time here, with
            # the canonicalised args, the second time directly in the
            # subclass.  But that one should hit the cache and return
            # straight away.
            obj.__init__(*args, **kwargs)
            # If key is None we're not supposed to store the object in cache
            if key:
                cls._cache_store(key, obj)
            return obj

    @classmethod
    def _cache_lookup(cls, key):
        return cls._cache[key]

    @classmethod
    def _cache_store(cls, key, val):
        cls._cache[key] = val

    @classmethod
    def _process_args(cls, *args, **kwargs):
        """Pre-processes the arguments before they are being passed to
        :meth:`_cache_key` and the constructor.

        :rtype: *must* return a :class:`list` of *args* and a
            :class:`dict` of *kwargs*"""
        return args, kwargs

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        """Compute the cache key given the preprocessed constructor arguments.

        :rtype: Cache key to use or ``None`` if the object is not to be cached

        .. note:: The cache key must be hashable."""
        return tuple(args) + tuple([(k, v) for k, v in kwargs.items()])

    @property
    def cache_key(self):
        """Cache key."""
        return self._key


class KernelCached(Cached):

    """Base class providing functionalities for cachable kernel objects."""

    def __new__(cls, *args, **kwargs):
        args, kwargs = cls._process_args(*args, **kwargs)
        code = cls._ast_to_c(*args, **kwargs)
        args = (code,) + args[1:]
        obj = super(KernelCached, cls).__new__(cls, *args, **kwargs)
        return obj

    @classmethod
    def _ast_to_c(cls, ast, name, opts={}, include_dirs=[]):
        """Transform an Abstract Syntax Tree representing the kernel into a
        string of C code."""
        if isinstance(ast, Node):
            return ast.gencode()
        return ast


class DiskCached(Cached):

    """Base class providing global caching of objects on disk. The same notes
    as in :class:`Cached` apply. In addition, derived classes need to
    define a class attribute :attr:`_cachedir` specifying the path where to
    cache objects on disk.

    .. warning ::
        The key returned by :meth:`_cache_key` *must* be a
        :class:`str` safe to use as a filename, such as an md5 hex digest.
    """

    @classmethod
    def _cache_lookup(cls, key):
        return cls._cache.get(key) or cls._read_from_disk(key)

    @classmethod
    def _read_from_disk(cls, key):
        filepath = os.path.join(cls._cachedir, key)
        if os.path.exists(filepath):
            f = gzip.open(filepath, "rb")
            val = cPickle.load(f)
            f.close()
            # Store in memory so we can save ourselves a disk lookup next time
            cls._cache[key] = val
            return val
        raise KeyError("Object with key %s not found in %s" % (key, filepath))

    @classmethod
    def _cache_store(cls, key, val):
        cls._cache[key] = val
        f = gzip.open(os.path.join(cls._cachedir, key), "wb")
        cPickle.dump(val, f)
        f.close()


class Versioned(object):
    """Versioning class for objects with mutable data"""

    def __new__(cls, *args, **kwargs):
        obj = super(Versioned, cls).__new__(cls)
        obj._version = 1
        obj._version_before_zero = 1
        #obj.__init__(*args, **kwargs)
        return obj

    def vcache_get_version(self):
        return self._version

    def vcache_version_bump(self):
        self._version_before_zero += 1
        # Undo version = 0
        self._version = self._version_before_zero

    def vcache_version_set_zero(self):
        # Set version to 0 (usually when zero() is called)
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

        self.vcache_version_bump()

        return retval

    return inner


def modifies_arguments(func):
    "Decorator for functions that modify their arguments' data"
    def inner(*args, **kwargs):
        retval = func(*args, **kwargs)
        for a in args:
            if hasattr(a, 'access') and a.access != op2.READ:
                a.data.vcache_version_bump()
        return retval
    return inner


class CopyOnWrite(object):
    """
    Class that overrides the copy and duplicate methods and performs the actual
    copy operation when either the original or the copy has been written.
    Classes that inherit from CopyOnWrite need to provide the methods:

    _cow_actual_copy(self, src):
        Performs an actual copy of src's data to self

    (optionally, otherwise copy.copy() is used)
    _cow_shallow_copy(self):

        Returns a shallow copy of the current object, e.g. the data handle
        should be the same
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


class AssemblyCache(object):
    """Singleton class"""
    _instance = None

    class CacheEntry(object):
        def __init__(self, form_sig, obj, dependencies=tuple()):
            self.form_sig = form_sig
            self.dependencies = dependencies
            self.obj = obj.duplicate()

        def is_valid(self):
            return all([d.is_valid() for d in self.dependencies])

        def get_object(self):
            return self.obj

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(AssemblyCache, cls).__new__(cls)
            cls._instance._hits = 0
            cls._instance._hits_size = 0
            cls._instance._enabled = True
            cls._instance.cache = {}
        return cls._instance

    def lookup(self, form_sig):
        cache_entry = self.cache.get(form_sig, None)

        retval = None
        if cache_entry is not None:
            if not cache_entry.is_valid():
                del self.cache[form_sig]
                return None

            retval = cache_entry.get_object()
            self._hits += 1

            debug('Object %s was retrieved from cache' % retval)
            debug('%d objects in cache' % self.num_objects)
        return retval

    def store(self, form_sig, obj, dependencies):
        cache_entry = AssemblyCache.CacheEntry(form_sig, obj, dependencies)
        self.cache[form_sig] = cache_entry

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    @property
    def num_objects(self):
        return len(self.cache.keys())

    @classmethod
    def cache_stats(cls):
        stats = "OpCache statistics: \n"
        stats += "\tnum_stored=%d\tbytes=%d \thits=%d\thit_bytes=%d" % \
            (self.num_objects, self.nbytes, self._hits, self._hits_size)
        return stats

    @property
    def nbytes(self):
        #TODO: DataCarrier subtypes should provide a 'bytes' property
        tot_bytes = 0
        for entry in self.cache.values():
            entry.get_object()
            tot_bytes += item.nbytes
        return tot_bytes


def assembly_cache(func):
    def inner(form):
        cache = AssemblyCache()
        form_sig = compute_form_signature(form)
        obj = cache.lookup(form_sig)
        if obj:
            print "Cache hit"
            return obj
        print "Cache miss"

        fd = form.compute_form_data()
        coords = form.integrals()[0].measure().domain_data()
        args = [coords]
        for c in fd.original_coefficients:
            args.append(c.dat(c.cell_dof_map, op2.READ))

        dependencies = tuple([arg.create_snapshot() for arg in args if arg is not None])
        obj = func(form)
        cache.store(form_sig, obj, dependencies)
        return obj

    return inner


from ufl import *


def test_assembler():
    op2.init()

    E = FiniteElement("Lagrange", "triangle", 1)

    v = TestFunction(E)
    u = TrialFunction(E)
    a = v*u*dx

    m = assembler(a)
    return m


@assembly_cache
def assembler(form):
    from pyop2.ffc_interface import compile_form
    import numpy as np

    # Generate code for mass and rhs assembly.

    mass, = compile_form(form, "mass")

    # Set up simulation data structures

    NUM_ELE = 2
    NUM_NODES = 4
    valuetype = np.float64

    nodes = op2.Set(NUM_NODES, 1, "nodes")
    vnodes = op2.Set(NUM_NODES, 2, "vnodes")
    elements = op2.Set(NUM_ELE, 1, "elements")

    elem_node_map = np.asarray([0, 1, 3, 2, 3, 1], dtype=np.uint32)
    elem_node = op2.Map(elements, nodes, 3, elem_node_map, "elem_node")
    elem_vnode = op2.Map(elements, vnodes, 3, elem_node_map, "elem_vnode")

    sparsity = op2.Sparsity((elem_node, elem_node), "sparsity")
    mat = op2.Mat(sparsity, valuetype, "mat")

    coord_vals = np.asarray([(0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.5)],
                            dtype=valuetype)
    coords = op2.Dat(vnodes, coord_vals, valuetype, "coords")

    # Assemble and solve

    op2.par_loop(mass, elements(3, 3),
                 mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                 coords(elem_vnode, op2.READ))
    return mat
