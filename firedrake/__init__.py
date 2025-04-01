def init_petsc4py():
    import configparser
    import os
    import pathlib
    import sys

    try:
        import petsc4py  # noqa: F401
        return
    except ImportError:
        pass

    config = configparser.ConfigParser()
    dir = pathlib.Path(__file__).parent
    with open(dir / "config.ini", "r") as f:
        config.read_file(f)

    petsc_dir = config["settings"]["petsc_dir"]
    petsc_arch = config["settings"]["petsc_arch"]
    sys.path.insert(0, os.path.join(petsc_dir, petsc_arch, "lib"))

    try:
        import petsc4py  # noqa: F401, F811
    except ImportError:
        raise Exception("can't find petsc4py, bad install?")


init_petsc4py()

from firedrake.configuration import setup_cache_dirs

# Set up the cache directories before importing PyOP2.
setup_cache_dirs()

# Ensure petsc is initialised by us before anything else gets in there.
# We conditionally pass '-options_left no' as in some circumstances (e.g.
# when running pytest) PETSc complains that command line options are not
# PETSc options.
import os
import sys
import petsc4py
if os.getenv("FIREDRAKE_DISABLE_OPTIONS_LEFT") == "1":
    petsc4py.init(sys.argv + ["-options_left", "no"])
else:
    petsc4py.init(sys.argv)
del os, sys, petsc4py

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
import sys
sys.modules["firedrake.plot"] = plot
from firedrake.plot import *

check()
del check, sys

from firedrake._version import get_versions
__version__ = get_versions()['version']
del get_versions


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
    from firedrake.petsc import get_blas_library

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

from . import _version
__version__ = _version.get_versions()['version']
