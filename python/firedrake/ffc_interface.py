"""Provides the interface to FFC for compiling a form, and transforms the FFC-
generated code in order to make it suitable for passing to the backends."""

from hashlib import md5
import os
import tempfile

from ufl import Form
from ufl.algorithms import as_form
from ffc import default_parameters, compile_form as ffc_compile_form
from ffc import constants
from ffc.log import set_level, ERROR

from pyop2.caching import DiskCached, KernelCached
from pyop2.op2 import Kernel
from pyop2.mpi import MPI
from pyop2.ir.ast_base import PreprocessNode, Root

_form_cache = {}

# Silence FFC
set_level(ERROR)

ffc_parameters = default_parameters()
ffc_parameters['write_file'] = False
ffc_parameters['format'] = 'pyop2'
ffc_parameters['pyop2-ir'] = True

# Include an md5 hash of firedrake_geometry.h in the cache key
with open(os.path.join(os.path.dirname(__file__), 'firedrake_geometry.h')) as f:
    _firedrake_geometry_md5 = md5(f.read()).hexdigest()


def _check_version():
    from pyop2.version import __compatible_ffc_version_info__ as compatible_version, \
        __compatible_ffc_version__ as version
    try:
        if constants.PYOP2_VERSION_INFO[:2] == compatible_version[:2]:
            return
    except AttributeError:
        pass
    raise RuntimeError("Incompatible PyOP2 version %s and FFC PyOP2 version %s."
                       % (version, getattr(constants, 'PYOP2_VERSION', 'unknown')))


class FFCKernel(DiskCached, KernelCached):

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

        form_data = form.form_data()

        kernels = []
        for ida, kernel in zip(form_data.integral_data, ffc_tree):
            # Set optimization options
            opts = {} if ida.domain_type not in ['cell'] else \
                   {'licm': False,
                    'tile': None,
                    'vect': None,
                    'ap': False}
            kernels.append(Kernel(Root([incl, kernel]), '%s_%s_integral_0_%s' %
                          (name, ida.domain_type, ida.domain_id), opts, inc))
        self.kernels = tuple(kernels)

        self._initialized = True


def compile_form(form, name):
    """Compile a form using FFC and return a :class:`pyop2.op2.Kernel`."""

    # Check that we get a Form
    if not isinstance(form, Form):
        form = as_form(form)

    return FFCKernel(form, name).kernels


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
