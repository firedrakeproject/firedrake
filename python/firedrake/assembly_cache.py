from collections import defaultdict
import ufl
from ufl.algorithms.signature import compute_form_signature
from pyop2 import op2
from pyop2.logger import warning, RED
import firedrake


class CacheEntry(object):
    """This is the basic caching unit. The form signature forms the key for
    each CacheEntry, while a reference to the main data object is kept.
    Additionally a list of Snapshot objects are kept in self.dependencies that
    together form a snapshot of all the data objects used during assembly.

    The validity of each CacheEntry object depends on the validity of its
    dependencies (i.e., that none of the referred objects have changed)."""

    def __init__(self, form_sig, obj, form, dependencies=tuple()):
        self.form_sig = form_sig
        self.form = form
        self.dependencies = dependencies
        self.obj = obj.duplicate()

    def is_valid(self):
        return all([d.is_valid() for d in self.dependencies])

    def get_object(self):
        return self.obj


class FragmentedCacheEntry(object):
    """A FragmentedCacheEntry is a combination of partial forms that will
    yield the original form requested.

    This implementation is a stub for the basic case of Product/Sum splitting
    of forms"""

    def __init__(self, partial_forms):
        self.partial_forms = partial_forms

    def get_object(self):
        sum_parts = [p.get_object() for p in self.partial_forms]

        #XXX sum() won't work because Dats don't implement __radd__
        return reduce((lambda x, y: x+y), sum_parts)

    def is_valid(self):
        return True


def ufl_flatten(op):
    """A function that returns a 'flat' form of the given UFL operator, i.e. a
    product operator will distribute the product to each of the inner sums"""

    if isinstance(op, ufl.algebra.Product):
        left = op.operands()[0]
        right = op.operands()[1]

        if isinstance(left, ufl.algebra.Sum):
            left_left = ufl_flatten(left.operands()[0])
            left_right = ufl_flatten(left.operands()[1])
            return ufl_flatten(right*left_left) + ufl_flatten(right*left_right)

        if isinstance(right, ufl.algebra.Sum):
            right_left = ufl_flatten(right.operands()[0])
            right_right = ufl_flatten(right.operands()[1])
            return ufl_flatten(left*right_left) + ufl_flatten(left*right_right)

        return left*right
    return op


def ufl_split_sums(op):

    """Simple function that breaks a UFL Sum tree to a list of forms, each
    element corresponding to an operand of the original long sum."""

    if isinstance(op, ufl.algebra.Sum):
        left = op.operands()[0]
        right = op.operands()[1]
        return ufl_split_sums(left) + ufl_split_sums(right)
    return [op]


class PartialForm(object):
    """PartialForm class implementing the behaviour of the partial forms
    expected by FragmentedCacheEntry"""

    #TODO Handle other cases
    def __init__(self, form, coefficient):
        self.form = form
        self.coefficient = coefficient

    def get_object(self):
        #print "Getting object for form %s, coefficient %s" % (self.form, self.coefficient)
        cache = AssemblyCache()
        obj = cache.lookup(self.form)
        if not obj:
            obj = cache.store(self.form)

        result = obj

        if self.coefficient:
            fd = self.form.compute_form_data()
            test = fd.original_arguments[0]
            result = test.make_dat()
            result.zero()
            #TODO PyOP2 Mat objects should provide an interface for numerical operations
            obj.handle.mult(self.coefficient.dat.vec, result.vec)

        return result

    def __repr__(self):
        return repr(self.form) + "-->(*%s) " % self.coefficient

    def __str(self):
        return "%s" % self.form, "-->(*%s) " % self.coefficient


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
            cls._instance._enabled = True
            cls._instance.cache = {}
            cls._instance.invalid_count = defaultdict(int)
            cls._instance.do_not_cache = set()
            cls._instance.assemblyfunc = None
        return cls._instance

    def lookup(self, form):
        form_sig = compute_form_signature(form)
        cache_entry = self.cache.get(form_sig, None)

        retval = None
        if cache_entry is not None:
            if not cache_entry.is_valid():
                self.invalid_count[form_sig] += 1
                del self.cache[form_sig]
                return None

            retval = cache_entry.get_object()
            self._hits += 1
            self._hits_size += retval.nbytes

            #debug.deprint('Object %s was retrieved from cache' % retval)
            #debug.deprint('%d objects in cache' % self.num_objects)
        return retval

    def _store_fragmented(self, form):
        form_sig = compute_form_signature(form)
        #TODO: multiple integrals?
        expr = form.integrals()[0].integrand()
        expr = ufl_flatten(expr)
        expr = ufl_split_sums(expr)
        forms = [e*firedrake.dx for e in expr]

        partial_forms = []

        for f in forms:
            fd = f.compute_form_data()
            # Can only factor out one coefficient
            volatile_coefficient = None
            highest_version = 1
            # Choose the one with the max version
            for c in fd.original_coefficients:
                if c.dat.vcache_get_version() > highest_version:
                    volatile_coefficient = c
                    highest_version = c.dat.vcache_get_version()
            function_space = volatile_coefficient.function_space()
            u = firedrake.TrialFunction(function_space)

            pf = f
            if highest_version > 2:
                pf = ufl.algorithms.replace(f, {volatile_coefficient: u})
            else:
                volatile_coefficient = None
            partial_forms.append(PartialForm(pf, volatile_coefficient))

        frag = FragmentedCacheEntry(partial_forms)
        self.cache[form_sig] = frag
        return frag.get_object()

    def store(self, form):
        form_sig = compute_form_signature(form)
        fd = form.compute_form_data()

        if self.invalid_count[form_sig] > 1:
            #print "Storing fragmented form"
            return self._store_fragmented(form)

        coords = form.integrals()[0].measure().domain_data()
        # XXX clean this up
        if hasattr(coords, 'dat'):
            args = [coords.dat]
        else:
            args = [coords]
        for c in fd.original_coefficients:
            args.append(c.dat(c.cell_dof_map, op2.READ))

        obj = self.assemblyfunc(form)

        dependencies = tuple()
        for arg in args:
            if arg is None:
                continue

            # XXX clean this up (maybe)
            if hasattr(arg, 'create_snapshot'):
                dep = arg.create_snapshot()
            elif hasattr(arg.data, 'create_snapshot'):
                dep = arg.data.create_snapshot()
            else:
                warning(RED, "Can't create snapshot for argument %r\n\
                             Object %r will not be cached" %
                        (arg, obj))
                return obj

            dependencies += (dep,)

        cache_entry = CacheEntry(form_sig, obj, form, dependencies)
        self.cache[form_sig] = cache_entry

        #for d in dependencies:
        #    print d._original(), d._snapshot_version
        return obj

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

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
        #TODO: DataCarrier subtypes should provide an 'nbytes' property
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


def assembly_cache(func):
    """This is the decorator interface to the assembly cache. It expects an
    assemble() type function that returns the assembled data object."""

    def inner(form):
        cache = AssemblyCache()
        if not cache.assemblyfunc:
            cache.assemblyfunc = func

        if not cache.enabled:
            return func(form)

        #form_sig = compute_form_signature(form)
        #if cache.invalid_count[form_sig] > 3:
        #    cache.do_not_cache.add(form_sig)
        #    return func(form)

        obj = cache.lookup(form)
        #debug.deprint(cache.cache_stats())
        if obj:
            #print "Cache hit"
            return obj

        #print "Cache miss"
        #from IPython import embed; embed()

        obj = cache.store(form)
        return obj

    return inner
