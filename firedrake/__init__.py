# The range of PETSc versions supported by Firedrake. Note that unlike in
# firedrake-configure and pyproject.toml where we want to be strict about
# the specific version, here we are more permissive. This is to catch the
# case where users don't update their PETSc for a really long time or
# accidentally install a too-new release that isn't yet supported.
PETSC_SUPPORTED_VERSIONS = ">=3.23.0"


def init_petsc():
    import os
    import sys
    import petsctools

    # We conditionally pass '-options_left no' as in some circumstances (e.g.
    # when running pytest) PETSc complains that command line options are not
    # PETSc options.
    if os.getenv("FIREDRAKE_DISABLE_OPTIONS_LEFT") == "1":
        petsctools.init(sys.argv + ["-options_left", "no"], version_spec=PETSC_SUPPORTED_VERSIONS)
    else:
        petsctools.init(sys.argv, version_spec=PETSC_SUPPORTED_VERSIONS)


# Ensure petsc is initialised right away
init_petsc()

# Set up the cache directories before importing PyOP2.
from firedrake.configuration import setup_cache_dirs

setup_cache_dirs()


# Initialise PETSc events for both import and entire duration of program
import petsctools
from firedrake import petsc
_is_logging = "log_view" in petsctools.OptionsManager.commandline_options
if _is_logging:
    _main_event = petsc.PETSc.Log.Event("firedrake")
    _main_event.begin()

    _init_event = petsc.PETSc.Log.Event("firedrake.__init__")
    _init_event.begin()

    import atexit
    atexit.register(lambda: _main_event.end())
    del atexit
del petsctools
del petsc

from ufl import *
from finat.ufl import *

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
import sys
sys.modules["firedrake.plot"] = plot
from firedrake.plot import *

del sys


def set_blas_num_threads():
    """Try to detect threading and either disable or warn user.

    Threading may come from
    - OMP_NUM_THREADS: openmp,
    - OPENBLAS_NUM_THREADS: openblas,
    - MKL_NUM_THREADS: mkl,
    - VECLIB_MAXIMUM_THREADS: accelerate,
    - NUMEXPR_NUM_THREADS: numexpr
    We only handle the first three cases

    """
    from ctypes import cdll
    from petsctools import get_blas_library

    try:
        blas_lib_path = get_blas_library()
    except:  # noqa: E722
        info("Cannot detect BLAS library, not setting the thread count")
        return

    try:
        blas_lib = cdll.LoadLibrary(blas_lib_path)
        method = None
        if "openblas" in blas_lib_path:
            method = "openblas_set_num_threads"
        elif "libmkl" in blas_lib_path:
            method = "MKL_Set_Num_Threads"

        if method:
            try:
                getattr(blas_lib, method)(1)
            except AttributeError:
                info("Cannot set number of threads in BLAS library")
    except OSError:
        info("Cannot set number of threads in BLAS library because the library could not be loaded")
    except TypeError:
        info("Cannot set number of threads in BLAS library because the library could not be found")


set_blas_num_threads()
del set_blas_num_threads


def warn_omp_num_threads():
    import os

    if os.getenv("OMP_NUM_THREADS") != "1":
        warning("OMP_NUM_THREADS is not set or is set to a value greater than 1, "
                "we suggest setting OMP_NUM_THREADS=1 to improve performance")


warn_omp_num_threads()
del warn_omp_num_threads

# Stop profiling Firedrake import
if _is_logging:
    _init_event.end()
    del _init_event
del _is_logging
