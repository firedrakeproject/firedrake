"""Provides the interface to FFC for compiling a form, and transforms the FFC-
generated code in order to make it suitable for passing to the backends."""

from collections import defaultdict
from hashlib import md5
from operator import add
from os import path, environ, getuid, makedirs
import tempfile

from ufl import Form, FiniteElement, VectorElement, as_vector
from ufl.algorithms import as_form, ReuseTransformer
from ufl.constantvalue import Zero
from ufl_expr import Argument

from ffc import default_parameters, compile_form as ffc_compile_form
from ffc import constants
from ffc import log

from pyop2.caching import DiskCached
from pyop2.op2 import Kernel
from pyop2.mpi import MPI
from pyop2.coffee.ast_base import PreprocessNode, Root

import functionspace

_form_cache = {}

ffc_parameters = default_parameters()
ffc_parameters['write_file'] = False
ffc_parameters['format'] = 'pyop2'
ffc_parameters['pyop2-ir'] = True

# Only spew ffc message on rank zero
if MPI.comm.rank != 0:
    log.set_level(log.ERROR)
del log


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
        args = tuple((a.count(), len(a.function_space()))
                     for a in form.form_data().original_arguments)
        forms_list = []
        for it in sum_integrands(form).integrals():
            forms = []

            def visit(idx):
                integrand = self.visit(it.integrand())
                if not isinstance(integrand, Zero):
                    forms.append([(idx, integrand * it.measure())])
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
                a = Argument(fs.ufl_element(), fs, arg.count())
                if a.shape():
                    if self._idx[arg.count()] == i:
                        args += [a[j] for j in range(a.shape()[0])]
                    else:
                        args += [Zero() for j in range(a.shape()[0])]
                else:
                    if self._idx[arg.count()] == i:
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
    def _cache_key(cls, form, name):
        form_data = form.compute_form_data()
        return md5(form_data.signature + name + Kernel._backend.__name__ +
                   cls._firedrake_geometry_md5 + constants.FFC_VERSION +
                   constants.PYOP2_VERSION).hexdigest()

    def __init__(self, form, name):
        """A wrapper object for one or more FFC kernels compiled from a given :class:`~Form`.

        :arg form: the :class:`~Form` from which to compile the kernels.
        :arg name: a prefix to be applied to the compiled kernel names. This is primarily useful for debugging.
        """
        if self._initialized:
            return

        incl = PreprocessNode('#include "firedrake_geometry.h"\n')
        inc = [path.dirname(__file__)]
        ffc_tree = ffc_compile_form(form, prefix=name, parameters=ffc_parameters)

        kernels = []
        for it, kernel in zip(form.form_data().preprocessed_form.integrals(), ffc_tree):
            # Set optimization options
            opts = {} if it.domain_type() not in ['cell'] else \
                   {'licm': False,
                    'slice': None,
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

_check_version()
_ensure_cachedir()
