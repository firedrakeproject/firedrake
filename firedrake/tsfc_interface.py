"""Provides the interface to TSFC for compiling a form, and transforms the TSFC-
generated code in order to make it suitable for passing to the backends."""
import pickle

from hashlib import md5
from os import path, environ, getuid, makedirs
import gzip
import os
import zlib
import tempfile
import collections

import ufl
from ufl import Form
from .ufl_expr import TestFunction

from tsfc import compile_form as tsfc_compile_form
from tsfc.parameters import PARAMETERS as tsfc_default_parameters

from pyop2.caching import Cached
from pyop2.op2 import Kernel
from pyop2.mpi import COMM_WORLD

from coffee.base import Invert

from firedrake.formmanipulation import split_form

from firedrake.parameters import parameters as default_parameters
from firedrake import utils


# Set TSFC default scalar type at load time
tsfc_default_parameters["scalar_type"] = utils.ScalarType_c


KernelInfo = collections.namedtuple("KernelInfo",
                                    ["kernel",
                                     "integral_type",
                                     "oriented",
                                     "subdomain_id",
                                     "domain_number",
                                     "coefficient_map",
                                     "needs_cell_facets",
                                     "pass_layer_arg",
                                     "needs_cell_sizes"])


class TSFCKernel(Cached):

    _cache = {}

    _cachedir = environ.get('FIREDRAKE_TSFC_KERNEL_CACHE_DIR',
                            path.join(tempfile.gettempdir(),
                                      'firedrake-tsfc-kernel-cache-uid%d' % getuid()))

    @classmethod
    def _cache_lookup(cls, key):
        key, comm = key
        return cls._cache.get(key) or cls._read_from_disk(key, comm)

    @classmethod
    def _read_from_disk(cls, key, comm):
        if comm.rank == 0:
            cache = cls._cachedir
            shard, disk_key = key[:2], key[2:]
            filepath = os.path.join(cache, shard, disk_key)
            val = None
            if os.path.exists(filepath):
                try:
                    with gzip.open(filepath, 'rb') as f:
                        val = f.read()
                except zlib.error:
                    pass

            comm.bcast(val, root=0)
        else:
            val = comm.bcast(None, root=0)

        if val is None:
            raise KeyError("Object with key %s not found" % key)
        return cls._cache.setdefault(key, pickle.loads(val))

    @classmethod
    def _cache_store(cls, key, val):
        key, comm = key
        cls._cache[key] = val
        _ensure_cachedir(comm=comm)
        if comm.rank == 0:
            val._key = key
            shard, disk_key = key[:2], key[2:]
            filepath = os.path.join(cls._cachedir, shard, disk_key)
            tempfile = os.path.join(cls._cachedir, shard, "%s_p%d.tmp" % (disk_key, os.getpid()))
            # No need for a barrier after this, since non root
            # processes will never race on this file.
            os.makedirs(os.path.join(cls._cachedir, shard), exist_ok=True)
            with gzip.open(tempfile, 'wb') as f:
                pickle.dump(val, f, 0)
            os.rename(tempfile, filepath)
        comm.barrier()

    @classmethod
    def _cache_key(cls, form, name, parameters, number_map, interface, coffee=False):
        # FIXME Making the COFFEE parameters part of the cache key causes
        # unnecessary repeated calls to TSFC when actually only the kernel code
        # needs to be regenerated
        return md5((form.signature() + name
                    + str(sorted(default_parameters["coffee"].items()))
                    + str(sorted(parameters.items()))
                    + str(number_map)
                    + str(type(interface))
                    + str(coffee)).encode()).hexdigest(), form.ufl_domains()[0].comm

    def __init__(self, form, name, parameters, number_map, interface, coffee=False):
        """A wrapper object for one or more TSFC kernels compiled from a given :class:`~ufl.classes.Form`.

        :arg form: the :class:`~ufl.classes.Form` from which to compile the kernels.
        :arg name: a prefix to be applied to the compiled kernel names. This is primarily useful for debugging.
        :arg parameters: a dict of parameters to pass to the form compiler.
        :arg number_map: a map from local coefficient numbers to global ones (useful for split forms).
        :arg interface: the KernelBuilder interface for TSFC (may be None)
        """
        if self._initialized:
            return

        assemble_inverse = parameters.get("assemble_inverse", False)
        coffee = coffee or assemble_inverse
        tree = tsfc_compile_form(form, prefix=name, parameters=parameters, interface=interface, coffee=coffee)
        kernels = []
        for kernel in tree:
            # Set optimization options
            opts = default_parameters["coffee"]
            ast = kernel.ast
            ast = ast if not assemble_inverse else _inverse(ast)
            # Unwind coefficient numbering
            numbers = tuple(number_map[c] for c in kernel.coefficient_numbers)
            kernels.append(KernelInfo(kernel=Kernel(ast, ast.name, opts=opts),
                                      integral_type=kernel.integral_type,
                                      oriented=kernel.oriented,
                                      subdomain_id=kernel.subdomain_id,
                                      domain_number=kernel.domain_number,
                                      coefficient_map=numbers,
                                      needs_cell_facets=False,
                                      pass_layer_arg=False,
                                      needs_cell_sizes=kernel.needs_cell_sizes))
        self.kernels = tuple(kernels)
        self._initialized = True


SplitKernel = collections.namedtuple("SplitKernel", ["indices",
                                                     "kinfo"])


def compile_form(form, name, parameters=None, inverse=False, split=True, interface=None, coffee=False):
    """Compile a form using TSFC.

    :arg form: the :class:`~ufl.classes.Form` to compile.
    :arg name: a prefix for the generated kernel functions.
    :arg parameters: optional dict of parameters to pass to the form
         compiler. If not provided, parameters are read from the
         ``form_compiler`` slot of the Firedrake
         :data:`~.parameters` dictionary (which see).
    :arg inverse: If True then assemble the inverse of the local tensor.
    :arg split: If ``False``, then don't split mixed forms.
    :arg coffee: compile coffee kernel instead of loopy kernel

    Returns a tuple of tuples of
    (index, integral type, subdomain id, coordinates, coefficients, needs_orientations, :class:`Kernels <pyop2.op2.Kernel>`).

    ``needs_orientations`` indicates whether the form requires cell
    orientation information (for correctly pulling back to reference
    elements on embedded manifolds).

    The coordinates are extracted from the domain of the integral (a
    :func:`~.Mesh`)

    """

    # Check that we get a Form
    if not isinstance(form, Form):
        raise RuntimeError("Unable to convert object to a UFL form: %s" % repr(form))

    if parameters is None:
        parameters = default_parameters["form_compiler"].copy()
    else:
        # Override defaults with user-specified values
        _ = parameters
        parameters = default_parameters["form_compiler"].copy()
        parameters.update(_)

    # We stash the compiled kernels on the form so we don't have to recompile
    # if we assemble the same form again with the same optimisations
    cache = form._cache.setdefault("firedrake_kernels", {})

    def tuplify(params):
        return tuple((k, params[k]) for k in sorted(params))

    key = (tuplify(default_parameters["coffee"]), name, tuplify(parameters), split)
    try:
        return cache[key]
    except KeyError:
        pass

    kernels = []
    # A map from all form coefficients to their number.
    coefficient_numbers = dict((c, n)
                               for (n, c) in enumerate(form.coefficients()))
    if split:
        iterable = split_form(form)
    else:
        iterable = ([(0, )*len(form.arguments()), form], )
    for idx, f in iterable:
        f = _real_mangle(f)
        # Map local coefficient numbers (as seen inside the
        # compiler) to the global coefficient numbers
        number_map = dict((n, coefficient_numbers[c])
                          for (n, c) in enumerate(f.coefficients()))
        kinfos = TSFCKernel(f, name + "".join(map(str, idx)), parameters,
                            number_map, interface, coffee).kernels
        for kinfo in kinfos:
            kernels.append(SplitKernel(idx, kinfo))
    kernels = tuple(kernels)
    return cache.setdefault(key, kernels)


def _real_mangle(form):
    """If the form contains arguments in the Real function space, replace these with literal 1 before passing to tsfc."""

    a = form.arguments()
    reals = [x.ufl_element().family() == "Real" for x in a]
    if not any(reals):
        return form
    replacements = {}
    for arg, r in zip(a, reals):
        if r:
            replacements[arg] = 1
    # If only the test space is Real, we need to turn the trial function into a test function.
    if reals == [True, False]:
        replacements[a[1]] = TestFunction(a[1].function_space())
    return ufl.replace(form, replacements)


def clear_cache(comm=None):
    """Clear the Firedrake TSFC kernel cache."""
    comm = comm or COMM_WORLD
    if comm.rank == 0:
        import shutil
        shutil.rmtree(TSFCKernel._cachedir, ignore_errors=True)
        _ensure_cachedir(comm=comm)


def _ensure_cachedir(comm=None):
    """Ensure that the TSFC kernel cache directory exists."""
    comm = comm or COMM_WORLD
    if comm.rank == 0:
        makedirs(TSFCKernel._cachedir, exist_ok=True)


def _inverse(kernel):
    """Modify ``kernel`` so to assemble the inverse of the local tensor."""

    local_tensor = kernel.args[0]

    if len(local_tensor.size) != 2 or local_tensor.size[0] != local_tensor.size[1]:
        raise ValueError("Can only assemble the inverse of a square 2-form")

    name = local_tensor.sym.symbol
    size = local_tensor.size[0]

    kernel.children[0].children.append(Invert(name, size))

    return kernel
