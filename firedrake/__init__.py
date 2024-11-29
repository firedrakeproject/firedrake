import os
import sys
from firedrake.configuration import setup_cache_dirs

# Set up the cache directories before importing PyOP2.
setup_cache_dirs()

# Ensure petsc is initialised by us before anything else gets in there.
#
# When running with pytest-xdist (i.e. pytest -n <#procs>) PETSc finalize will
# crash (see https://github.com/firedrakeproject/firedrake/issues/3247). This
# is because PETSc wants to complain about unused options to stderr, but by this
# point the worker's stderr stream has already been destroyed by xdist, causing
# a crash. To prevent this we disable unused options checking in PETSc when
# running with xdist.
import petsc4py
if "PYTEST_XDIST_WORKER" in os.environ:
    petsc4py.init(sys.argv + ["-options_left", "no"])
else:
    petsc4py.init(sys.argv)
del petsc4py

# Initialise PETSc events for both import and entire duration of program
from firedrake import petsc
_is_logging = "log_view" in petsc.OptionsManager.commandline_options
if _is_logging:
    _main_event = petsc.PETSc.Log.Event("firedrake")
    _main_event.begin()

    _init_event = petsc.PETSc.Log.Event("firedrake.__init__")
    _init_event.begin()

    import atexit
    atexit.register(lambda: _main_event.end())
    del atexit

_blas_lib_path = petsc.get_blas_library()
del petsc

# UFL Exprs come with a custom __del__ method, but we hold references
# to them /everywhere/, some of which are circular (the Mesh object
# holds a ufl.Domain that references the Mesh).  The Python2 GC
# explicitly DOES NOT collect such reference cycles (even though it
# can deal with normal cycles).  Quoth the documentation:
#
#     Objects that have __del__() methods and are part of a reference
#     cycle cause the entire reference cycle to be uncollectable,
#     including objects not necessarily in the cycle but reachable
#     only from it.
#
# To get around this, since the default __del__ on Expr is just
# "pass", we just remove the method from the definition of Expr.
import ufl
try:
    del ufl.core.expr.Expr.__del__
except AttributeError:
    pass
del ufl
from ufl import *
from finat.ufl import *

# By default we disable pyadjoint annotation.
# To enable annotation, the user has to call continue_annotation().
import pyadjoint
pyadjoint.pause_annotation()
del pyadjoint

from firedrake_citations import Citations    # noqa: F401
# Always get the firedrake paper.
Citations().register("FiredrakeUserManual")
from pyop2 import op2                        # noqa: F401
from pyop2.mpi import COMM_WORLD, COMM_SELF  # noqa: F401

from firedrake.assemble import *
from firedrake.bcs import *
from firedrake.checkpointing import *
from firedrake.cofunction import *
from firedrake.constant import *
from firedrake.exceptions import *
from firedrake.function import *
from firedrake.functionspace import *
from firedrake.interpolation import *
from firedrake.linear_solver import *
from firedrake.preconditioners import *
from firedrake.mesh import *
from firedrake.mg.mesh import *
from firedrake.mg.interface import *
from firedrake.mg.embedded import *
from firedrake.mg.opencascade_mh import *
from firedrake.norms import *
from firedrake.nullspace import *
from firedrake.optimizer import *
from firedrake.parameters import *
from firedrake.parloops import *
from firedrake.projection import *
from firedrake.slate import *
from firedrake.slope_limiter import *
from firedrake.solving import *
from firedrake.ufl_expr import *
from firedrake.utility_meshes import *
from firedrake.variational_solver import *
from firedrake.eigensolver import *
from firedrake.vector import *
from firedrake.version import __version__ as ver, __version_info__, check  # noqa: F401
from firedrake.ensemble import *
from firedrake.randomfunctiongen import *
from firedrake.external_operators import *
from firedrake.progress_bar import ProgressBar  # noqa: F401

from firedrake.logging import *
# Set default log level
set_log_level(WARNING)
set_log_handlers(comm=COMM_WORLD)

# Moved functionality
from firedrake._deprecation import plot, File  # noqa: F401
# Once `File` is deprecated update the above line removing `File` and add
#   from firedrake._deprecation import output
#   sys.modules["firedrake.output"] = output
from firedrake.output import *
sys.modules["firedrake.plot"] = plot
from firedrake.plot import *

check()
del check, sys

from firedrake._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Try to detect threading and either disable or warn user
# Threading may come from
# - OMP_NUM_THREADS: openmp,
# - OPENBLAS_NUM_THREADS: openblas,
# - MKL_NUM_THREADS: mkl,
# - VECLIB_MAXIMUM_THREADS: accelerate,
# - NUMEXPR_NUM_THREADS: numexpr
# We only handle the first three cases
from ctypes import cdll
try:
    _blas_lib = cdll.LoadLibrary(_blas_lib_path)
    _method_name = None
    if "openblas" in _blas_lib_path:
        _method_name = "openblas_set_num_threads"
    elif "libmkl" in _blas_lib_path:
        _method_name = "MKL_Set_Num_Threads"

    if _method_name:
        try:
            getattr(_blas_lib, _method_name)(1)
        except AttributeError:
            info("Cannot set number of threads in BLAS library")
except OSError:
    info("Cannot set number of threads in BLAS library because the library could not be loaded")
except TypeError:
    info("Cannot set number of threads in BLAS library because the library could not be found")

# OMP_NUM_THREADS can be set to a comma-separated list of positive integers
try:
    _omp_num_threads = int(os.environ.get('OMP_NUM_THREADS'))
except (ValueError, TypeError):
    _omp_num_threads = None
if (_omp_num_threads is None) or (_omp_num_threads > 1):
    warning('OMP_NUM_THREADS is not set or is set to a value greater than 1,'
            ' we suggest setting OMP_NUM_THREADS=1 to improve performance')
del _blas_lib, _method_name, _omp_num_threads, os, cdll

# Stop profiling Firedrake import
if _is_logging:
    _init_event.end()
    del _init_event
del _is_logging

from . import _version
__version__ = _version.get_versions()['version']
