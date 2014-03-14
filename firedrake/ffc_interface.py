"""Provides the interface to FFC for compiling a form, and transforms the FFC-
generated code in order to make it suitable for passing to the backends."""

from collections import defaultdict
from hashlib import md5
from operator import add
import os
import tempfile

from ufl import Form, FiniteElement, VectorElement, as_vector
from ufl.algorithms import as_form, ReuseTransformer
from ufl.constantvalue import Zero
from ufl_expr import Argument

from ffc import default_parameters, compile_form as ffc_compile_form
from ffc import constants

from pyop2.caching import DiskCached
from pyop2.op2 import Kernel
from pyop2.mpi import MPI
from pyop2.ir.ast_base import PreprocessNode, Root

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
    integrals = defaultdict(list)
    for integral in form.integrals():
        integrals[integral.measure()].append(integral)
    return Form([it[0].reconstruct(reduce(add, [i.integrand() for i in it]))
                 for it in integrals.values()])


class FormSplitter(ReuseTransformer):
    """Split a form into a subtree for each component of the mixed space it is
    built on. This is a no-op on forms over non-mixed spaces."""

    def split(self, form):
        """Split the given form."""
        # Visit each integrand and obtain the tuple of sub forms
        shape = tuple(len(a.function_space())
                      for a in form.form_data().original_arguments)
        forms_list = []
        for it in sum_integrands(form).integrals():
            forms = []
            for i in range(shape[0] if len(shape) > 0 else 1):
                for j in range(shape[1] if len(shape) > 1 else 1):
                    self._idx = {-2: i, -1: j}
                    integrand = self.visit(it.integrand())
                    if not isinstance(integrand, Zero):
                        forms.append([((i, j), integrand * it.measure())])
            forms_list += forms
        return forms_list

    def argument(self, o):
        """Split an argument into its constituent spaces."""
        if isinstance(o.function_space(), types.MixedFunctionSpace):
            args = []
            for i, fs in enumerate(o.function_space().split()):
                a = Argument(fs.ufl_element(), fs, o.count())
                if a.shape():
                    if self._idx[o.count()] == i:
                        args += [a[j] for j in range(a.shape()[0])]
                    else:
                        args += [Zero() for j in range(a.shape()[0])]
                else:
                    if self._idx[o.count()] == i:
                        args.append(a)
                    else:
                        args.append(Zero())
            return as_vector(args)
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
        for it, kernel in zip(form.form_data().preprocessed_form.integrals(), ffc_tree):
            # Set optimization options
            opts = {} if it.domain_type() not in ['cell'] else \
                   {'licm': False,
                    'tile': None,
                    'vect': None,
                    'ap': False,
                    'split': None}
            kernels.append(Kernel(Root([incl, kernel]), '%s_%s_integral_0_%s' %
                           (name, it.domain_type(), it.domain_id()), opts, inc))
        self.kernels = tuple(kernels)
        self._initialized = True


def compile_form(form, name):
    """Compile a form using FFC and return a tuple of tuples of
    (index, domain type, coefficients, :class:`Kernels <pyop2.op2.Kernel>`)."""

    # Check that we get a Form
    if not isinstance(form, Form):
        form = as_form(form)

    fd = form.compute_form_data()
    # If there is no mixed element involved, return the kernels FFC produces
    if all(isinstance(e, (FiniteElement, VectorElement)) for e in fd.unique_sub_elements):
        return [((0, 0), it.measure(), fd.original_coefficients, kernel)
                for it, kernel in zip(fd.preprocessed_form.integrals(),
                                      FFCKernel(form, name).kernels)]
    # Otherwise pre-split the form into mixed blocks before calling FFC
    kernels = []
    for forms in FormSplitter().split(form):
        for (i, j), form in forms:
            kernel, = FFCKernel(form, name + str(i) + str(j)).kernels
            fd = form.form_data()
            kernels.append(((i, j), fd.preprocessed_form.integrals()[0].measure(),
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
