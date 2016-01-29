"""Provides the interface to FFC for compiling a form, and transforms the FFC-
generated code in order to make it suitable for passing to the backends."""
from __future__ import absolute_import

from collections import defaultdict
from hashlib import md5
from operator import add
from os import path, environ, getuid, makedirs
import tempfile

from ufl import Form, as_vector
from ufl.measure import Measure
from ufl.algorithms import ReuseTransformer
from ufl.constantvalue import Zero
from firedrake.ufl_expr import Argument

from firedrake.fc import compile_form as ffc_compile_form

from pyop2.caching import DiskCached
from pyop2.op2 import Kernel
from pyop2.mpi import MPI

from coffee.base import Invert

import firedrake.functionspace as functionspace
from firedrake.parameters import parameters as default_parameters


def sum_integrands(form):
    """Produce a form with the integrands on the same measure summed."""
    integrals = defaultdict(list)
    for integral in form.integrals():
        md = integral.metadata()
        mdkey = tuple((k, md[k]) for k in sorted(md.keys()))
        integrals[(integral.integral_type(),
                   integral.ufl_domain(),
                   integral.subdomain_data(),
                   integral.subdomain_id(),
                   mdkey)].append(integral)
    return Form([it[0].reconstruct(reduce(add, [i.integrand() for i in it]))
                 for it in integrals.values()])


class FormSplitter(ReuseTransformer):
    """Split a form into a subtree for each component of the mixed space it is
    built on. This is a no-op on forms over non-mixed spaces."""

    def split(self, form):
        """Split the given form."""
        args = form.arguments()
        if not any(isinstance(a.function_space(), functionspace.MixedFunctionSpace)
                   for a in args):
            return [[((0, 0), form)]]
        # Visit each integrand and obtain the tuple of sub forms
        args = tuple((a.number(), len(a.function_space()))
                     for a in form.arguments())
        forms_list = []
        for it in sum_integrands(form).integrals():
            forms = []

            def visit(idx):
                integrand = self.visit(it.integrand())
                if not isinstance(integrand, Zero):
                    forms.append([(idx, integrand * Measure(it.integral_type(),
                                                            domain=it.ufl_domain(),
                                                            subdomain_id=it.subdomain_id(),
                                                            subdomain_data=it.subdomain_data(),
                                                            metadata=it.metadata()))])
            # 0 form
            if not args:
                visit((0, 0))
            # 1 form
            elif len(args) == 1:
                count, l = args[0]
                for i in range(l):
                    self._idx = {count: i}
                    self._args = {}
                    visit((i, 0))
            # 2 form
            elif len(args) == 2:
                for i in range(args[0][1]):
                    for j in range(args[1][1]):
                        self._idx = {args[0][0]: i, args[1][0]: j}
                        self._args = {}
                        visit((i, j))
            forms_list += forms
        return forms_list

    def argument(self, arg):
        """Split an argument into its constituent spaces."""
        from itertools import product
        if isinstance(arg.function_space(), functionspace.MixedFunctionSpace):
            # Look up the split argument in cache since we want it unique
            if arg in self._args:
                return self._args[arg]
            args = []
            for i, fs in enumerate(arg.function_space().split()):
                # Build the sub-space Argument (not part of the mixed
                # space).
                a = Argument(fs, arg.number(), part=arg.part())

                # Produce indexing iterator.
                # For scalar-valued spaces this results in the empty
                # tuple (), which returns the Argument when indexing.
                # For vector-valued spaces, this is just
                # range(a.ufl_shape[0])
                # For tensor-valued spaces, it's the nested loop of
                # range(x for x in a.ufl_shape).
                #
                # Each of the indexed things is then scalar-valued
                # which we turn into a ufl Vector.
                indices = product(*map(range, a.ufl_shape))
                if self._idx[arg.number()] == i:
                    args += [a[idx] for idx in indices]
                else:
                    args += [Zero() for _ in indices]
            self._args[arg] = as_vector(args)
            return self._args[arg]
        return arg


class FFCKernel(DiskCached):

    _cache = {}
    if MPI.comm.rank == 0:
        _cachedir = environ.get('FIREDRAKE_FFC_KERNEL_CACHE_DIR',
                                path.join(tempfile.gettempdir(),
                                          'firedrake-ffc-kernel-cache-uid%d' % getuid()))
    else:
        _cachedir = None

    @classmethod
    def _cache_key(cls, form, name, parameters, number_map):
        # FIXME Making the COFFEE parameters part of the cache key causes
        # unnecessary repeated calls to FFC when actually only the kernel code
        # needs to be regenerated
        return md5(form.signature() + name + Kernel._backend.__name__
                   + str(default_parameters["coffee"])
                   + str(parameters)
                   + str(number_map)).hexdigest()

    def __init__(self, form, name, parameters, number_map):
        """A wrapper object for one or more FFC kernels compiled from a given :class:`~ufl.classes.Form`.

        :arg form: the :class:`~ufl.classes.Form` from which to compile the kernels.
        :arg name: a prefix to be applied to the compiled kernel names. This is primarily useful for debugging.
        :arg parameters: a dict of parameters to pass to the form compiler.
        :arg number_map: a map from local coefficient numbers to global ones (useful for split forms).
        """
        if self._initialized:
            return

        ffc_tree = ffc_compile_form(form, prefix=name, parameters=parameters)
        kernels = []
        for kernel in ffc_tree:
            # Set optimization options
            opts = default_parameters["coffee"]
            ast = kernel.ast
            ast = ast if not parameters.get("assemble_inverse", False) else _inverse(ast)
            # Unwind coefficient numbering
            numbers = tuple(number_map[c] for c in kernel.coefficient_numbers)
            kernels.append((Kernel(ast, ast.name, opts=opts),
                            kernel.integral_type,
                            kernel.oriented,
                            kernel.subdomain_id,
                            numbers))
        self.kernels = tuple(kernels)
        self._initialized = True


def compile_form(form, name, parameters=None, inverse=False):
    """Compile a form using FFC.

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
        # Save both kernels and FFC params so we can tell if this
        # cached version is valid (the FFC parameters might have changed)
        kernels, coffee_params, old_name, params = form._cache["firedrake_kernels"]
        if coffee_params == default_parameters["coffee"] and \
           name == old_name and \
           params == parameters:
            return kernels

    kernels = []
    # A map from all form coefficients to their number.
    coefficient_numbers = dict((c, n)
                               for (n, c) in enumerate(form.coefficients()))
    for forms in FormSplitter().split(form):
        for (i, j), f in forms:
            # Map local coefficient numbers (as seen inside the
            # compiler) to the global coefficient numbers
            number_map = dict((n, coefficient_numbers[c])
                              for (n, c) in enumerate(f.coefficients()))
            ffc_kernel = FFCKernel(f, name + str(i) + str(j), parameters,
                                   number_map)
            for kinfo in ffc_kernel.kernels:
                kernels.append(((i, j),
                                kinfo))
    kernels = tuple(kernels)
    form._cache["firedrake_kernels"] = (kernels, default_parameters["coffee"].copy(),
                                        name, parameters)
    return kernels


def clear_cache():
    """Clear the Firedrake FFC kernel cache."""
    if MPI.comm.rank != 0:
        return
    if path.exists(FFCKernel._cachedir):
        import shutil
        shutil.rmtree(FFCKernel._cachedir, ignore_errors=True)
        _ensure_cachedir()


def _ensure_cachedir():
    """Ensure that the FFC kernel cache directory exists."""
    if MPI.comm.rank != 0:
        return
    if not path.exists(FFCKernel._cachedir):
        makedirs(FFCKernel._cachedir)


def _inverse(kernel):
    """Modify ``kernel`` so to assemble the inverse of the local tensor."""

    local_tensor = kernel.args[0]

    if len(local_tensor.size) != 2 or local_tensor.size[0] != local_tensor.size[1]:
        raise ValueError("Can only assemble the inverse of a square 2-form")

    name = local_tensor.sym.symbol
    size = local_tensor.size[0]

    kernel.children[0].children.append(Invert(name, size))

    return kernel


_ensure_cachedir()
