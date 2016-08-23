"""Provides the interface to TSFC for compiling a form, and transforms the TSFC-
generated code in order to make it suitable for passing to the backends."""
from __future__ import absolute_import

from hashlib import md5
from os import path, environ, getuid, makedirs
import cPickle
import gzip
import os
import zlib
import tempfile
import numpy
import collections

from ufl import Form, as_vector
from ufl.corealg.map_dag import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.constantvalue import Zero
from firedrake.ufl_expr import Argument

from tsfc import compile_form as tsfc_compile_form

from pyop2.caching import Cached
from pyop2.op2 import Kernel
from pyop2.mpi import COMM_WORLD, dup_comm, free_comm

from coffee.base import Invert

from firedrake.parameters import parameters as default_parameters


SplitForm = collections.namedtuple("SplitForm", ["indices", "form"])


class FormSplitter(MultiFunction):

    """Split a form in a list of subtrees for each component of the
    mixed space it is built on.  See :meth:`split` for a usage
    description."""

    def split(self, form):
        """Split the form.

        :arg form: the form to split.

        This is a no-op if none of the arguments in the form are
        defined on :class:`~.MixedFunctionSpace`\s.

        The return-value is a tuple for which each entry is.

        .. code-block:: python

           (argument_indices, form)

        Where ``argument_indices`` is a tuple indicating which part of
        the mixed space the form belongs to, it has length equal to
        the number of arguments in the form.  Hence functionals have
        a 0-tuple, 1-forms have a 1-tuple and 2-forms a 2-tuple
        of indices.

        For example, consider the following code:

        .. code-block:: python

            V = FunctionSpace(m, 'CG', 1)
            W = V*V*V
            u, v, w = TrialFunctions(W)
            p, q, r = TestFunctions(W)
            a = q*u*dx + p*w*dx

        Then splitting the form returns a tuple of two forms.

        .. code-block:: python

           ((0, 2), w*p*dx),
            (1, 0), q*u*dx))

        """
        args = form.arguments()
        if all(len(a.function_space()) == 1 for a in args):
            # No mixed spaces, just return the form directly.
            idx = tuple([0]*len(form.arguments()))
            return (SplitForm(indices=idx, form=form), )
        forms = []
        # How many subspaces do we have for each argument?
        shape = tuple(len(a.function_space()) for a in args)
        # Walk over all the indices of the spaces
        for idx in numpy.ndindex(shape):
            # Which subspace are we currently interested in?
            self.idx = dict(enumerate(idx))
            # Cache for the arguments we construct
            self._args = {}
            # Visit the form
            f = map_integrand_dags(self, form)
            # Zero-simplification may result in an empty form, only
            # collect those that are non-zero.
            if len(f.integrals()) > 0:
                forms.append(SplitForm(indices=idx, form=f))
        return tuple(forms)

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        return o

    def argument(self, o):
        V = o.function_space()
        if len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o
        # Already seen this argument, return the cached version.
        if o in self._args:
            return self._args[o]
        args = []
        for i, V_i in enumerate(V.split()):
            # Walk over the subspaces and build a vector that is zero
            # where the argument does not match the one we're looking
            # for and is just the non-mixed argument when we do want
            # it.
            a = Argument(V_i, o.number(), part=o.part())
            indices = numpy.ndindex(a.ufl_shape)
            if self.idx[o.number()] == i:
                args += [a[j] for j in indices]
            else:
                args += [Zero() for j in indices]
        self._args[o] = as_vector(args)
        return self._args[o]


KernelInfo = collections.namedtuple("KernelInfo",
                                    ["kernel",
                                     "integral_type",
                                     "oriented",
                                     "subdomain_id",
                                     "domain_number",
                                     "coefficient_map"])


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
            filepath = os.path.join(cache, key)
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
        val = cPickle.loads(val)
        cls._cache[key] = val
        return val

    @classmethod
    def _cache_store(cls, key, val):
        key, comm = key
        _ensure_cachedir(comm=comm)
        if comm.rank == 0:
            val._key = key
            filepath = os.path.join(cls._cachedir, key)
            tempfile = os.path.join(cls._cachedir, "%s_p%d.tmp" % (key, os.getpid()))
            # No need for a barrier after this, since non root
            # processes will never race on this file.
            with gzip.open(tempfile, 'wb') as f:
                cPickle.dump(val, f)
            os.rename(tempfile, filepath)
        comm.barrier()

    @classmethod
    def _cache_key(cls, form, name, parameters, number_map):
        # FIXME Making the COFFEE parameters part of the cache key causes
        # unnecessary repeated calls to TSFC when actually only the kernel code
        # needs to be regenerated
        return md5(form.signature() + name + Kernel._backend.__name__
                   + str(default_parameters["coffee"])
                   + str(parameters)
                   + str(number_map)).hexdigest(), form.ufl_domains()[0].comm

    def __init__(self, form, name, parameters, number_map):
        """A wrapper object for one or more TSFC kernels compiled from a given :class:`~ufl.classes.Form`.

        :arg form: the :class:`~ufl.classes.Form` from which to compile the kernels.
        :arg name: a prefix to be applied to the compiled kernel names. This is primarily useful for debugging.
        :arg parameters: a dict of parameters to pass to the form compiler.
        :arg number_map: a map from local coefficient numbers to global ones (useful for split forms).
        """
        if self._initialized:
            return

        tree = tsfc_compile_form(form, prefix=name, parameters=parameters)
        kernels = []
        for kernel in tree:
            # Set optimization options
            opts = default_parameters["coffee"]
            ast = kernel.ast
            ast = ast if not parameters.get("assemble_inverse", False) else _inverse(ast)
            # Unwind coefficient numbering
            numbers = tuple(number_map[c] for c in kernel.coefficient_numbers)
            kernels.append(KernelInfo(kernel=Kernel(ast, ast.name, opts=opts),
                                      integral_type=kernel.integral_type,
                                      oriented=kernel.oriented,
                                      subdomain_id=kernel.subdomain_id,
                                      domain_number=kernel.domain_number,
                                      coefficient_map=numbers))
        self.kernels = tuple(kernels)
        self._initialized = True


SplitKernel = collections.namedtuple("SplitKernel", ["indices",
                                                     "kinfo"])


def compile_form(form, name, parameters=None, inverse=False):
    """Compile a form using TSFC.

    :arg form: the :class:`~ufl.classes.Form` to compile.
    :arg name: a prefix for the generated kernel functions.
    :arg parameters: optional dict of parameters to pass to the form
         compiler. If not provided, parameters are read from the
         ``form_compiler`` slot of the Firedrake
         :data:`~.parameters` dictionary (which see).
    :arg inverse: If True then assemble the inverse of the local tensor.

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
    if "firedrake_kernels" in form._cache:
        # Save both kernels and TSFC params so we can tell if this
        # cached version is valid (the TSFC parameters might have changed)
        kernels, coffee_params, old_name, params = form._cache["firedrake_kernels"]
        if coffee_params == default_parameters["coffee"] and \
           name == old_name and \
           params == parameters:
            return kernels

    kernels = []
    # A map from all form coefficients to their number.
    coefficient_numbers = dict((c, n)
                               for (n, c) in enumerate(form.coefficients()))
    for idx, f in FormSplitter().split(form):
        # Map local coefficient numbers (as seen inside the
        # compiler) to the global coefficient numbers
        number_map = dict((n, coefficient_numbers[c])
                          for (n, c) in enumerate(f.coefficients()))
        kinfos = TSFCKernel(f, name + "".join(map(str, idx)), parameters,
                            number_map).kernels
        for kinfo in kinfos:
            kernels.append(SplitKernel(idx, kinfo))
    kernels = tuple(kernels)
    form._cache["firedrake_kernels"] = (kernels, default_parameters["coffee"].copy(),
                                        name, parameters)
    return kernels


def clear_cache(comm=None):
    """Clear the Firedrake TSFC kernel cache."""
    comm = dup_comm(comm or COMM_WORLD)
    if comm.rank == 0:
        if path.exists(TSFCKernel._cachedir):
            import shutil
            shutil.rmtree(TSFCKernel._cachedir, ignore_errors=True)
            _ensure_cachedir(comm=comm)
    free_comm(comm)


def _ensure_cachedir(comm=None):
    """Ensure that the TSFC kernel cache directory exists."""
    comm = dup_comm(comm or COMM_WORLD)
    if comm.rank == 0:
        if not path.exists(TSFCKernel._cachedir):
            makedirs(TSFCKernel._cachedir)
    free_comm(comm)


def _inverse(kernel):
    """Modify ``kernel`` so to assemble the inverse of the local tensor."""

    local_tensor = kernel.args[0]

    if len(local_tensor.size) != 2 or local_tensor.size[0] != local_tensor.size[1]:
        raise ValueError("Can only assemble the inverse of a square 2-form")

    name = local_tensor.sym.symbol
    size = local_tensor.size[0]

    kernel.children[0].children.append(Invert(name, size))

    return kernel
