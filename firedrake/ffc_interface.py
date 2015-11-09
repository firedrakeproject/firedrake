"""Provides the interface to FFC for compiling a form, and transforms the FFC-
generated code in order to make it suitable for passing to the backends."""
from __future__ import absolute_import

from collections import defaultdict
from hashlib import md5
from operator import add
from os import path, environ, getuid, makedirs
import tempfile

import ufl
from ufl import Form, MixedElement, as_vector
from ufl.measure import Measure
from ufl.algorithms import compute_form_data, ReuseTransformer
from ufl.constantvalue import Zero
from firedrake.ufl_expr import Argument

from ffc import compile_form as ffc_compile_form
from ffc import constants
from ffc import log
from ffc.quadrature.quadraturetransformerbase import EmptyIntegrandError

from pyop2.caching import DiskCached
from pyop2.op2 import Kernel
from pyop2.mpi import MPI

from coffee.base import PreprocessNode, Root, Invert

import firedrake.fiat_utils as fiat_utils
import firedrake.functionspace as functionspace
from firedrake.parameters import parameters as default_parameters

_form_cache = {}

# Only spew ffc message on rank zero
if MPI.comm.rank != 0:
    log.set_level(log.ERROR)
del log


def _check_version():
    from firedrake.version import __compatible_ffc_version_info__ as compatible_version, \
        __compatible_ffc_version__ as version
    try:
        if constants.FIREDRAKE_VERSION_INFO[:2] == compatible_version[:2]:
            return
    except AttributeError:
        pass
    raise RuntimeError("Incompatible Firedrake version %s and FFC version %s."
                       % (version, getattr(constants, 'FIREDRAKE_VERSION', 'unknown')))


def sum_integrands(form):
    """Produce a form with the integrands on the same measure summed."""
    integrals = defaultdict(list)
    for integral in form.integrals():
        md = integral.metadata()
        mdkey = tuple((k, md[k]) for k in sorted(md.keys()))
        integrals[(integral.integral_type(),
                   integral.domain(),
                   integral.subdomain_id(),
                   mdkey)].append(integral)
    return Form([it[0].reconstruct(reduce(add, [i.integrand() for i in it]))
                 for it in integrals.values()])


class FormSplitter(ReuseTransformer):
    """Split a form into a subtree for each component of the mixed space it is
    built on. This is a no-op on forms over non-mixed spaces."""

    def split(self, form):
        """Split the given form."""
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
                                                            domain=it.domain(),
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
        if isinstance(arg.function_space(), functionspace.MixedFunctionSpace):
            if arg in self._args:
                return self._args[arg]
            args = []
            for i, fs in enumerate(arg.function_space().split()):
                # Look up the split argument in cache since we want it unique
                a = Argument(fs, arg.number(), part=arg.part())
                if a.shape():
                    if self._idx[arg.number()] == i:
                        args += [a[j] for j in range(a.shape()[0])]
                    else:
                        args += [Zero() for j in range(a.shape()[0])]
                else:
                    if self._idx[arg.number()] == i:
                        args.append(a)
                    else:
                        args.append(Zero())
            self._args[arg] = as_vector(args)
            return as_vector(args)
        return arg


class FFCKernel(DiskCached):

    _cache = {}
    if MPI.comm.rank == 0:
        _cachedir = environ.get('FIREDRAKE_FFC_KERNEL_CACHE_DIR',
                                path.join(tempfile.gettempdir(),
                                          'firedrake-ffc-kernel-cache-uid%d' % getuid()))
        # Include an md5 hash of firedrake_geometry.h in the cache key
        with open(path.join(path.dirname(__file__), 'firedrake_geometry.h')) as f:
            _firedrake_geometry_md5 = md5(f.read()).hexdigest()
        del f
        MPI.comm.bcast(_firedrake_geometry_md5, root=0)
    else:
        # No cache on slave processes
        _cachedir = None
        # MD5 obtained by broadcast from root
        _firedrake_geometry_md5 = MPI.comm.bcast(None, root=0)

    @classmethod
    def _cache_key(cls, form, name, parameters):
        # FIXME Making the COFFEE parameters part of the cache key causes
        # unnecessary repeated calls to FFC when actually only the kernel code
        # needs to be regenerated
        return md5(form.signature() + name + Kernel._backend.__name__ +
                   cls._firedrake_geometry_md5 + constants.FFC_VERSION +
                   constants.FIREDRAKE_VERSION + str(default_parameters["coffee"])
                   + str(parameters)).hexdigest()

    def _needs_orientations(self, elements):
        for e in elements:
            cell = e.cell()
            if cell.topological_dimension() == cell.geometric_dimension():
                continue
            if isinstance(e, ufl.MixedElement) and e.family() != 'Real':
                if any("contravariant piola" in fiat_utils.fiat_from_ufl_element(s).mapping()
                       for s in e.sub_elements()):
                    return True
            else:
                if e.family() != 'Real' and \
                   "contravariant piola" in fiat_utils.fiat_from_ufl_element(e).mapping():
                    return True
        return False

    def __init__(self, form, name, parameters):
        """A wrapper object for one or more FFC kernels compiled from a given :class:`~Form`.

        :arg form: the :class:`~Form` from which to compile the kernels.
        :arg name: a prefix to be applied to the compiled kernel names. This is primarily useful for debugging.
        :arg parameters: a dict of parameters to pass to the form compiler.
        """
        if self._initialized:
            return

        incl = [PreprocessNode('#include "firedrake_geometry.h"\n')]
        inc = [path.dirname(__file__)]
        try:
            ffc_tree = ffc_compile_form(form, prefix=name, parameters=parameters)
            if len(ffc_tree) == 0:
                raise EmptyIntegrandError
            kernels = []
            # need compute_form_data here to get preproc form integrals
            fd = compute_form_data(form)
            elements = fd.elements
            needs_orientations = self._needs_orientations(elements)
            for it, kernel in zip(fd.preprocessed_form.integrals(), ffc_tree):
                # Set optimization options
                opts = default_parameters["coffee"]
                _kernel = kernel if not parameters.get("assemble_inverse", False) else _inverse(kernel)
                kernels.append((Kernel(Root(incl + [_kernel]), '%s_%s_integral_0_%s' %
                                       (name, it.integral_type(), it.subdomain_id()), opts, inc),
                                needs_orientations))
            self.kernels = tuple(kernels)
            self._empty = False
        except EmptyIntegrandError:
            # FFC noticed that the integrand was zero and simplified
            # it, catch this here and set a flag telling us to ignore
            # the kernel when returning it in compile_form
            self._empty = True
        self._initialized = True


def compile_form(form, name, parameters=None, inverse=False):
    """Compile a form using FFC.

    :arg form: the :class:`ufl.Form` to compile.
    :arg name: a prefix for the generated kernel functions.
    :arg parameters: optional dict of parameters to pass to the form
         compiler. If not provided, parameters are read from the
         :data:`form_compiler` slot of the Firedrake
         :data:`~.parameters` dictionary (which see).
    :arg inverse: If True then assemble the inverse of the local tensor.

    Returns a tuple of tuples of
    (index, integral type, subdomain id, coordinates, coefficients, needs_orientations, :class:`Kernels <pyop2.op2.Kernel>`).

    ``needs_orientations`` indicates whether the form requires cell
    orientation information (for correctly pulling back to reference
    elements on embedded manifolds).

    The coordinates are extracted from the UFL
    :class:`~ufl.domain.Domain` of the integral.

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
        kernels, params = form._cache["firedrake_kernels"]
        if kernels[0][-1]._opts == default_parameters["coffee"] and \
           kernels[0][-1].name.startswith(name) and \
           params == parameters:
            return kernels

    # need compute_form_data since we use preproc. form integrals later
    fd = compute_form_data(form)

    # If there is no mixed element involved, return the kernels FFC produces
    # Note: using type rather than isinstance because UFL's VectorElement,
    # TensorElement and OPVectorElement all inherit from MixedElement
    if not any(type(e) is MixedElement for e in fd.unique_sub_elements):
        kernels = [((0, 0),
                    it.integral_type(), it.subdomain_id(),
                    it.domain().coordinates(),
                    fd.preprocessed_form.coefficients(), needs_orientations, kernel)
                   for it, (kernel, needs_orientations) in zip(fd.preprocessed_form.integrals(),
                                                               FFCKernel(form, name,
                                                                         parameters).kernels)]
        form._cache["firedrake_kernels"] = (kernels, parameters)
        return kernels
    # Otherwise pre-split the form into mixed blocks before calling FFC
    kernels = []
    for forms in FormSplitter().split(form):
        for (i, j), f in forms:
            ffc_kernel = FFCKernel(f, name + str(i) + str(j), parameters)
            # FFC noticed the integrand was zero, so don't bother
            # using this kernel (it's invalid anyway)
            if ffc_kernel._empty:
                continue
            ((kernel, needs_orientations), ) = ffc_kernel.kernels
            # need compute_form_data here to get preproc integrals
            fd = compute_form_data(f)
            it = fd.preprocessed_form.integrals()[0]
            kernels.append(((i, j),
                            it.integral_type(),
                            it.subdomain_id(),
                            it.domain().coordinates(),
                            fd.preprocessed_form.coefficients(),
                            needs_orientations, kernel))
    form._cache["firedrake_kernels"] = (kernels, parameters)
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


_check_version()
_ensure_cachedir()
