"""Provides the interface to FFC for compiling a form, and transforms the FFC-
generated code in order to make it suitable for passing to the backends."""

from hashlib import md5
from operator import add
import os
import tempfile
import numpy as np

from ufl import Form, FiniteElement, VectorElement
from ufl.algorithms import as_form, traverse_terminals, ReuseTransformer
from ufl.indexing import FixedIndex, MultiIndex
from ufl_expr import Argument

from ffc import default_parameters, compile_form as ffc_compile_form
from ffc import constants

from pyop2.caching import DiskCached
from pyop2.op2 import Kernel
from pyop2.mpi import MPI
from pyop2.ir.ast_base import PreprocessNode, Root
from pyop2.utils import as_tuple

import types

_form_cache = {}

ffc_parameters = default_parameters()
ffc_parameters['write_file'] = False
ffc_parameters['format'] = 'pyop2'
ffc_parameters['pyop2-ir'] = True

# Include an md5 hash of firedrake_geometry.h in the cache key
with open(os.path.join(os.path.dirname(__file__), 'firedrake_geometry.h')) as f:
    _firedrake_geometry_md5 = md5(f.read()).hexdigest()


def _check_version():
    from version import __compatible_ffc_version_info__ as compatible_version, \
        __compatible_ffc_version__ as version
    try:
        if constants.PYOP2_VERSION_INFO[:2] == compatible_version[:2]:
            return
    except AttributeError:
        pass
    raise RuntimeError("Incompatible PyOP2 version %s and FFC PyOP2 version %s."
                       % (version, getattr(constants, 'PYOP2_VERSION', 'unknown')))


def sum_integrands(form):
    """Produce a form with the integrands on the same measure summed."""
    return Form([it[0].reconstruct(reduce(add, [i.integrand() for i in it]))
                 for d, it in form.integral_groups().items()])


class FormSplitter(ReuseTransformer):
    """Split a form into a subtree for each component of the mixed space it is
    built on. This is a no-op on forms over non-mixed spaces."""

    def split(self, form):
        """Split the given form."""
        fd = form.compute_form_data()
        # If there is no mixed element involved, return a form per integral
        if all(isinstance(e, (FiniteElement, VectorElement)) for e in fd.unique_sub_elements):
            return [[Form([i])] for i in sum_integrands(form).integrals()]
        # Otherwise visit each integrand and obtain the tuple of sub forms
        return [[f * i.measure() for f in as_tuple(self.visit(i.integrand()))]
                for i in sum_integrands(form).integrals()]

    def sum(self, o, l, r):
        """Take the sum of operands on the same block and return a tuple of
        partial sums for each block."""

        def find_idx(e):
            """Find the block index of an expression given by the indices of
            the function spaces of the arguments (test and trial function)."""
            row, col = None, None
            for t in traverse_terminals(e):
                if isinstance(t, Argument):
                    if t.count() == -2:  # Test function gives the row
                        row = t.function_space().index
                    elif t.count() == -1:  # Trial function gives the column
                        col = t.function_space().index
            return (row, col)

        as_list = lambda o: list(o) if isinstance(o, (list, tuple)) else [o]
        res = []
        # For each (index, argument) tuple in the left operand list, look for
        # a tuple with corresponding index in the right operand list. If
        # there is one, append the sum of the arguments with that index to the
        # results list, otherwise just the tuple from the left operand list
        l = as_list(l)
        r = as_list(r)
        idx_r = [find_idx(i) for i in r]
        # Go over all the operands in the left operand list
        for a, i in zip(l, [find_idx(i) for i in l]):
            # If there is any operand in the right operand list on the same
            # block, take their sum
            try:
                j = idx_r.index(i)
                idx_r.pop(j)
                res.append(o.reconstruct(a, r.pop(j)))
            # Otherwise just append the operand from the left operand list
            except ValueError:
                res.append(a)
        # All remaining tuples in the right operand list had no matches, so we
        # append them to the results list
        return tuple(res + r) if len(res + r) > 1 else (res + r)[0]

    def _binop(self, o, l, r):
        if isinstance(l, tuple) and isinstance(r, tuple):
            return tuple(o.reconstruct(op1, op2) for op1, op2 in zip(l, r))
        else:
            return o.reconstruct(l, r)

    def inner(self, o, l, r):
        """Reconstruct an inner product on each of the component spaces."""
        return self._binop(o, l, r)

    def product(self, o, l, r):
        """Reconstruct a product on each of the component spaces."""
        return self._binop(o, l, r)

    def dot(self, o, l, r):
        """Reconstruct a dot product on each of the component spaces."""
        return self._binop(o, l, r)

    def _index(self, o, arg, idx):
        """Reconstruct an index if the rank matches, otherwise yield the
        argument. If the argument is a tuple, go over each entry."""
        build = lambda a: o.reconstruct(a, idx) if a.rank() == len(idx.free_indices()) else a
        if isinstance(arg, tuple):
            return tuple(build(a) for a in arg)
        else:
            return build(arg)

    def index_sum(self, o, arg, idx):
        """Reconstruct an index sum on each of the component spaces."""
        build = lambda a: o.reconstruct(a, idx) if len(a.free_indices()) == len(idx.free_indices()) else a
        if isinstance(arg, tuple):
            return tuple(build(a) for a in arg)
        else:
            return build(arg)

    def indexed(self, o, arg, idx):
        """Apply fixed indices where they point on a scalar subspace.
        Reconstruct fixed indices on a component vector and any other index."""
        if isinstance(idx._indices[0], FixedIndex):
            # Find the element to which the FixedIndex points. We might deal
            # with coefficients on vector elements, in which case we need to
            # reconstruct the indexed with an adjusted index space. Otherwise
            # we can just return the coefficient.
            i = idx._indices[0]._value
            pos = 0
            for op in arg:
                # If the FixedIndex points at a scalar (shapeless) operand,
                # return it
                if not op.shape() and i == pos:
                    return op
                size = np.prod(op.shape() or 1)
                # If the FixedIndex points at a component of the current
                # operand, reconstruct an Indexed with an adjusted index space
                if i < pos + size:
                    return o.reconstruct(op, MultiIndex(FixedIndex(i - pos), {}))
                # Otherwise update the position in the index space
                pos += size
            raise NotImplementedError("No idea what to in %r with %r" % (o, arg))
        return self._index(o, arg, idx)

    def argument(self, o):
        """Split an argument into its constituent spaces."""
        if isinstance(o.function_space(), types.MixedFunctionSpace):
            return tuple(Argument(fs.ufl_element(), fs, o.count())
                         for fs in o.function_space().split())
        return o

    def coefficient(self, o):
        """Split a coefficient into its constituent spaces."""
        if isinstance(o.function_space(), types.MixedFunctionSpace):
            return o.split()
        return o


class FFCKernel(DiskCached):

    _cache = {}
    _cachedir = os.path.join(tempfile.gettempdir(),
                             'firedrake-ffc-kernel-cache-uid%d' % os.getuid())

    @classmethod
    def _cache_key(cls, form, name):
        form_data = form.compute_form_data()
        return md5(form_data.signature + name + Kernel._backend.__name__ +
                   _firedrake_geometry_md5 + constants.FFC_VERSION +
                   constants.PYOP2_VERSION).hexdigest()

    def __init__(self, form, name):
        if self._initialized:
            return

        incl = PreprocessNode('#include "firedrake_geometry.h"\n')
        inc = [os.path.dirname(__file__)]
        ffc_tree = ffc_compile_form(form, prefix=name, parameters=ffc_parameters)

        kernels = []
        for ida, kernel in zip(form.form_data().integral_data, ffc_tree):
            # Set optimization options
            opts = {} if ida.domain_type not in ['cell'] else \
                   {'licm': False,
                    'tile': None,
                    'vect': None,
                    'ap': False,
                    'split': None}
            kernels.append(Kernel(Root([incl, kernel]), '%s_%s_integral_0_%s' %
                           (name, ida.domain_type, ida.domain_id), opts, inc))
        self.kernels = tuple(kernels)
        self._initialized = True


def compile_form(form, name):
    """Compile a form using FFC and return a tuple of tuples of
    (index, domain type, coefficients, :class:`Kernels <pyop2.op2.Kernel>`)."""

    # Check that we get a Form
    if not isinstance(form, Form):
        form = as_form(form)

    kernels = []
    for forms in FormSplitter().split(form):
        for i, form in enumerate(forms):
            kernel, = FFCKernel(form, name + str(i)).kernels

            fd = form.form_data()
            ida = fd.integral_data[0]
            if len(forms) == 1 and fd.rank == 0:
                idx = (0, 0)
            else:
                t = tuple(a.function_space().index or 0
                          for a in fd.original_arguments) or (i, 0)
                idx = t if len(t) == 2 else t + (0,) * (2 - len(t))
            kernels.append((idx, ida.integrals[0].measure(),
                            fd.original_coefficients, kernel))
    return kernels


def clear_cache():
    """Clear the PyOP2 FFC kernel cache."""
    if MPI.comm.rank != 0:
        return
    if os.path.exists(FFCKernel._cachedir):
        import shutil
        shutil.rmtree(FFCKernel._cachedir, ignore_errors=True)
        _ensure_cachedir()


def _ensure_cachedir():
    """Ensure that the FFC kernel cache directory exists."""
    if not os.path.exists(FFCKernel._cachedir) and MPI.comm.rank == 0:
        os.makedirs(FFCKernel._cachedir)

_check_version()
_ensure_cachedir()
