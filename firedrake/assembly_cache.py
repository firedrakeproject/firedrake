from collections import defaultdict
from ufl.algorithms.signature import compute_form_signature
import types
import weakref
from petsc4py import PETSc
from parameters import parameters


class DependencySnapshot(object):
    """Record the dependencies of a form at a particular point in order to
    establish whether a cached form is valid."""

    def __init__(self, form):

        # For each dependency, we store a weak reference and the
        # current version number.
        ref = lambda dep: (weakref.ref(dep), dep.dat._version)

        deps = []

        coords = form.integrals()[0].measure().domain_data()
        deps.append(ref(coords))

        for c in form.compute_form_data().original_coefficients:
            deps.append(ref(c))

        self.dependencies = tuple(deps)

    def valid(self, form):
        """Check whether form is valid with respect to this dependency snapshot."""

        original_coords = self.dependencies[0][0]()
        if original_coords:
            coords = form.integrals()[0].measure().domain_data()
            if coords != original_coords or \
               coords.dat._version != self.dependencies[0][1]:
                return False

        deps = form.compute_form_data().original_coefficients

        for original_d, dep in zip(self.dependencies[1:], deps):
            original_dep = original_d[0]()
            if original_dep:
                if dep != original_dep or dep.dat._version != original_d[1]:
                    return False

        return True


class BCSnapshot(object):
    """Record the boundary conditions which were applied to a form."""

    def __init__(self, bcs):

        self.bcs = map(weakref.ref, bcs)

    def valid(self, bcs):

        if len(bcs) != len(self.bcs):
            return False

        for bc, wbc in zip(bcs, self.bcs):
            if bc != wbc():
                return False

        return True


class CacheEntry(object):
    """This is the basic caching unit. The form signature forms the key for
    each CacheEntry, while a reference to the main data object is kept.
    Additionally a list of Snapshot objects are kept in self.dependencies that
    together form a snapshot of all the data objects used during assembly.

    The validity of each CacheEntry object depends on the validity of its
    dependencies (i.e., that none of the referred objects have changed)."""

    def __init__(self, obj, form, bcs):
        self.form = form
        self.dependencies = DependencySnapshot(form)
        self.bcs = BCSnapshot(bcs)
        if isinstance(obj, float):
            self.obj = obj
        else:
            self.obj = obj.duplicate()

    def is_valid(self, form, bcs):
        return self.dependencies.valid(form) and self.bcs.valid(bcs)

    def get_object(self):
        return self.obj


class AssemblyCache(object):
    """Singleton class.

    This is the central point of the assembly cache subsystem. This is a
    Singleton object so all the stored cache entries will reside in the single
    instance object returned.

    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(AssemblyCache, cls).__new__(cls)
            cls._instance._hits = 0
            cls._instance._hits_size = 0
            cls._instance.cache = {}
            cls._instance.invalid_count = defaultdict(int)
            cls._instance.do_not_cache = set()
            cls._instance.assemblyfunc = None
        return cls._instance

    def lookup(self, form, bcs):
        form_sig = compute_form_signature(form)
        cache_entry = self.cache.get(form_sig, None)

        retval = None
        if cache_entry is not None:
            if not cache_entry.is_valid(form, bcs):
                self.invalid_count[form_sig] += 1
                del self.cache[form_sig]
                return None

            retval = cache_entry.get_object()
            self._hits += 1
            self._hits_size += retval.nbytes

        return retval

    def store(self, obj, form, bcs):
        form_sig = compute_form_signature(form)

        cache_entry = CacheEntry(obj, form, bcs)
        self.cache[form_sig] = cache_entry

        return obj

    @property
    def num_objects(self):
        return len(self.cache.keys())

    def cache_stats(self):
        stats = "OpCache statistics: \n"
        #from IPython import embed; embed()
        stats += "\tnum_stored=%d\tbytes=%d\trealbytes=%d\thits=%d\thit_bytes=%d" % \
                 (self.num_objects, self.nbytes, self.realbytes, self._hits,
                  self._hits_size)
        return stats

    @property
    def nbytes(self):
        tot_bytes = 0
        for entry in self.cache.values():
            obj = entry.get_object()
            tot_bytes += obj.nbytes
        return tot_bytes

    @property
    def realbytes(self):
        tot_bytes = 0
        for entry in self.cache.values():
            obj = entry.get_object()
            if not (hasattr(obj, "_cow_is_copy_of") and obj._cow_is_copy_of):
                tot_bytes += obj.nbytes
            # TODO: also count snapshot bytes
            #for dep in obj.dependencies:
            #    if dep._duplicate
        return tot_bytes


def cache_thunk(thunk, form, result):
    """Wrap thunk so that thunk is only executed if its target is not in
    the cache."""

    def inner(bcs):

        cache = AssemblyCache()

        if not parameters["assembly_cache"]["enabled"]:
            return thunk(bcs)

        obj = cache.lookup(form, bcs)
        if obj is not None:
            if isinstance(result, float):
                # 0-form case
                assert isinstance(obj, float)
                r = obj
            elif isinstance(result, types.Function):
                # 1-form
                result.dat = obj
                r = result
            elif isinstance(result, types.Matrix):
                # 2-form
                if obj.handle is not result._M.handle:
                    obj.handle.copy(result._M.handle,
                                    PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
                r = result
            else:
                raise TypeError("Unknown result type")
            return r

        r = thunk(bcs)
        if isinstance(r, float):
            # 0-form case
            cache.store(r, form, bcs)
        elif isinstance(r, types.Function):
            # 1-form
            cache.store(r.dat, form, bcs)
        elif isinstance(r, types.Matrix):
            # 2-form
            cache.store(r._M, form, bcs)
        else:
            raise TypeError("Unknown result type")
        return r

    return inner
