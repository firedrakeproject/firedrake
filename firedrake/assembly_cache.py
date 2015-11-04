"""Firedrake by default caches assembled forms. This means that it is
generally not necessary for users to manually lift assembly of linear
operators out of timestepping loops; the same performance benefit will
be realised automatically by the assembly cache.

In order to prevent the assembly cache leaking memory, a simple cache
eviction strategy is implemented. This is documented below. In
addition, the following parameters control the operation of the
assembly_cache:

:data:`parameters["assembly_cache"]["enabled"]`
  a boolean value used to disable the assembly cache if required.

:data:`parameters["assembly_cache"]["eviction"]`
  a boolean value used to disable the cache eviction
  strategy. Disabling cache eviction can lead to memory leaks so is
  discouraged in almost all circumstances.

:data:`parameters["assembly_cache"]["max_misses"]`
  attempting to cache objects whose inputs change every time they are
  assembled is a waste of memory. This parameter sets a maximum number
  of consecutive misses beyond which a form will be marked as
  uncachable.

:data:`parameters["assembly_cache"]["max_bytes"]`
  absolute limit on the size of the assembly cache in bytes. This
  defaults to :data:`float("inf")`.

:data:`parameters["assembly_cache"]["max_factor"]`
  limit on the size of the assembly cache relative to the amount of
  memory per core on the current system. This defaults to 0.6.
"""
from __future__ import absolute_import
import numpy as np
import weakref
from collections import defaultdict

from pyop2.logger import debug, warning
from pyop2.mpi import MPI, _MPI

from firedrake.parameters import parameters
from firedrake.petsc import PETSc

try:
    # Estimate the amount of memory per core may use.
    import psutil
    memory = np.array([psutil.virtual_memory().total/psutil.cpu_count()])
    if MPI.comm.size > 1:
        MPI.comm.Allreduce(_MPI.IN_PLACE, memory, _MPI.MIN)
except (ImportError, AttributeError):
    memory = None


class _DependencySnapshot(object):
    """Record the dependencies of a form at a particular point in order to
    establish whether a cached form is valid."""

    def __init__(self, form):

        # For each dependency, we store a weak reference and the
        # current version number.
        ref = lambda dep: (weakref.ref(dep), dep.dat._version)

        deps = []

        coords = form.integrals()[0].domain().coordinates()
        deps.append(ref(coords))

        for c in form.coefficients():
            deps.append(ref(c))

        self.dependencies = tuple(deps)

    def valid(self, form):
        """Check whether form is valid with respect to this dependency snapshot."""

        original_coords = self.dependencies[0][0]()
        if original_coords:
            coords = form.integrals()[0].domain().coordinates()
            if coords is not original_coords or \
               coords.dat._version != self.dependencies[0][1]:
                return False
        else:
            return False

        # Since UFL sorts the coefficients by count (creation index),
        # further sorting here is not required.
        deps = form.coefficients()

        for original_d, dep in zip(self.dependencies[1:], deps):
            original_dep = original_d[0]()
            if original_dep:
                if dep is not original_dep or dep.dat._version != original_d[1]:
                    return False
            else:
                return False

        return True


class _BCSnapshot(object):
    """Record the boundary conditions which were applied to a form."""

    def __init__(self, bcs):

        self.bcs = map(weakref.ref, bcs) if bcs is not None else None

    def valid(self, bcs):

        if len(bcs) != len(self.bcs):
            return False

        for bc, wbc in zip(bcs, self.bcs):
            if bc != wbc():
                return False

        return True


class _CacheEntry(object):
    """This is the basic caching unit. The form signature forms the key for
    each CacheEntry, while a reference to the main data object is kept.
    Additionally a list of Snapshot objects are kept in self.dependencies that
    together form a snapshot of all the data objects used during assembly.

    The validity of each CacheEntry object depends on the validity of its
    dependencies (i.e., that none of the referred objects have changed)."""

    def __init__(self, obj, form, bcs):
        self.form = form
        self.dependencies = _DependencySnapshot(form)
        self.bcs = _BCSnapshot(bcs)
        if isinstance(obj, float):
            self.obj = np.float64(obj)
        else:
            self.obj = obj.duplicate()

        global _assemble_count
        self._assemble_count += 1
        self.value = self._assemble_count
        if MPI.comm.size > 1:
            tmp = np.array([obj.nbytes])
            MPI.comm.Allreduce(_MPI.IN_PLACE, tmp, _MPI.MAX)
            self.nbytes = tmp[0]
        else:
            self.nbytes = obj.nbytes

    _assemble_count = 0

    def is_valid(self, form, bcs):
        return self.dependencies.valid(form) and self.bcs.valid(bcs)

    def get_object(self):
        return self.obj


class AssemblyCache(object):
    """This is the central point of the assembly cache subsystem. This is a
    singleton object so all the stored cache entries will reside in the single
    instance object returned.

    It is not usually necessary for users to access the
    :class:`AssemblyCache` object directly, but this may occassionally
    be useful when studying performance problems.
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(AssemblyCache, cls).__new__(cls)
            cls._instance._hits = 0
            cls._instance._hits_size = 0
            cls._instance.cache = {}
            cls._instance.invalid_count = defaultdict(int)
            cls._instance.evictwarned = False
        return cls._instance

    def _lookup(self, form, bcs, ffc_parameters):
        form_sig = form.signature()
        parms, cache_entry = self.cache.get(form_sig, (None, None))

        retval = None
        if cache_entry is not None:
            if parms != str(ffc_parameters) or not cache_entry.is_valid(form, bcs):
                self.invalid_count[form_sig] += 1
                del self.cache[form_sig]
                return None
            else:
                self.invalid_count[form_sig] = 0

            retval = cache_entry.get_object()
            self._hits += 1
            self._hits_size += retval.nbytes

        return retval

    def _store(self, obj, form, bcs, ffc_parameters):
        form_sig = form.signature()

        if self.invalid_count[form_sig] > parameters["assembly_cache"]["max_misses"]:
            if self.invalid_count[form_sig] == \
               parameters["assembly_cache"]["max_misses"] + 1:
                debug("form %s missed too many times, excluding from cache." % form)

        else:
            cache_entry = _CacheEntry(obj, form, bcs)
            self.cache[form_sig] = str(ffc_parameters), cache_entry
            self.evict()

    def evict(self):
        """Run the cache eviction algorithm. This works out the permitted
cache size and deletes objects until it is achieved. Cache values are
assumed to have a :attr:`value` attribute and eviction occurs in
increasing :attr:`value` order. Currently :attr:`value` is an index of
the assembly operation, so older operations are evicted first.

The cache will be evicted down to 90% of permitted size.

The permitted size is either the explicit
:data:`parameters["assembly_cache"]["max_bytes"]` or it is the amount of
memory per core scaled by :data:`parameters["assembly_cache"]["max_factor"]`
(by default the scale factor is 0.6).

In MPI parallel, the nbytes of each cache entry is set to the maximum
over all processes, while the available memory is set to the
minimum. This produces a conservative caching policy which is
guaranteed to result in the same evictions on each processor.

        """

        if not parameters["assembly_cache"]["eviction"]:
            return

        max_cache_size = min(parameters["assembly_cache"]["max_bytes"] or float("inf"),
                             (memory or float("inf"))
                             * parameters["assembly_cache"]["max_factor"]
                             )

        if max_cache_size == float("inf"):
            if not self.evictwarned:
                warning("No maximum assembly cache size. Install psutil >= 2.0.0 or risk leaking memory!")
                self.evictwarned = True
            return

        cache_size = self.nbytes
        if cache_size < max_cache_size:
            return

        debug("Cache eviction triggered. %s bytes in cache, %s bytes allowed" %
              (cache_size, max_cache_size))

        # Evict down to 90% full.
        bytes_to_evict = cache_size - 0.9 * max_cache_size

        sorted_cache = sorted(self.cache.items(), key=lambda x: x[1][1].value)

        nbytes = lambda x: x[1][1].nbytes

        candidates = []
        while bytes_to_evict > 0:
            next = sorted_cache.pop(0)
            candidates.append(next)
            bytes_to_evict -= nbytes(next)

        for c in reversed(candidates):
            if bytes_to_evict + nbytes(c) < 0:
                # We may have been overzealous.
                bytes_to_evict += nbytes(c)
            else:
                del self.cache[c[0]]

    def clear(self):
        """Clear the cache contents."""
        self.cache = {}
        self._hits = 0
        self._hits_size = 0
        self.invalid_count = defaultdict(int)

    @property
    def num_objects(self):
        """The number of objects currently in the cache."""
        return len(self.cache)

    @property
    def cache_stats(self):
        """Consolidated statistics for the cache contents"""
        stats = "OpCache statistics: \n"
        stats += "\tnum_stored=%d\tbytes=%d\trealbytes=%d\thits=%d\thit_bytes=%d" % \
                 (self.num_objects, self.nbytes, self.realbytes, self._hits,
                  self._hits_size)
        return stats

    @property
    def nbytes(self):
        """An estimate of the total number of bytes in the cached objects."""
        return sum([entry.nbytes for _, entry in self.cache.values()])

    @property
    def realbytes(self):
        """An estimate of the total number of bytes for which the cache holds
        the sole reference to an object."""
        tot_bytes = 0
        for _, entry in self.cache.values():
            obj = entry.get_object()
            if not (hasattr(obj, "_cow_is_copy_of") and obj._cow_is_copy_of):
                tot_bytes += entry.nbytes
        return tot_bytes


def _cache_thunk(thunk, form, result, form_compiler_parameters=None):
    """Wrap thunk so that thunk is only executed if its target is not in
    the cache."""
    from firedrake import function
    from firedrake import matrix

    if form_compiler_parameters is None:
        form_compiler_parameters = parameters["form_compiler"]

    def inner(bcs):

        cache = AssemblyCache()

        if not parameters["assembly_cache"]["enabled"]:
            return thunk(bcs)

        obj = cache._lookup(form, bcs, form_compiler_parameters)
        if obj is not None:
            if isinstance(result, float):
                # 0-form case
                assert isinstance(obj, float)
                r = obj
            elif isinstance(result, function.Function):
                # 1-form
                result.dat = obj
                r = result
            elif isinstance(result, matrix.Matrix):
                # 2-form
                if obj.handle is not result._M.handle:
                    obj.handle.copy(result._M.handle,
                                    PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                    # Ensure result matrix is assembled (MatCopy_Nest bug)
                    if not result._M.handle.assembled:
                        result._M.handle.assemble()
                r = result
            else:
                raise TypeError("Unknown result type")
            return r

        r = thunk(bcs)
        if isinstance(r, float):
            # 0-form case
            cache._store(r, form, bcs, form_compiler_parameters)
        elif isinstance(r, function.Function):
            # 1-form
            cache._store(r.dat, form, bcs, form_compiler_parameters)
        elif isinstance(r, matrix.Matrix):
            # 2-form
            cache._store(r._M, form, bcs, form_compiler_parameters)
        else:
            raise TypeError("Unknown result type")
        return r

    return inner
