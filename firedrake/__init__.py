# Don't reorder imports in this file because we have to be careful of side effects
# isort: skip_file

# The range of PETSc versions supported by Firedrake. Note that unlike in
# firedrake-configure and pyproject.toml where we want to be strict about
# the specific version, here we are more permissive. This is to catch the
# case where users don't update their PETSc for a really long time or
# accidentally install a too-new release that isn't yet supported.
# TODO RELEASE set to ">=3.25"
PETSC_SUPPORTED_VERSIONS = ">=3.23.0"


def _init_petsc():
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


# Ensure PETSc is initialised right away
_init_petsc()

# Set up the cache directories before importing PyOP2.
from firedrake.configuration import setup_cache_dirs

setup_cache_dirs()


# Initialise PETSc events for both import and entire duration of program
import petsctools

from firedrake import petsc

_is_logging = "log_view" in petsctools.get_commandline_options()
if _is_logging:
    _main_event = petsc.PETSc.Log.Event("firedrake")
    _main_event.begin()

    _init_event = petsc.PETSc.Log.Event("firedrake.__init__")
    _init_event.begin()

    import atexit
    atexit.register(lambda: _main_event.end())
    del atexit
del petsc

# Register possible citations
import firedrake.citations  # noqa: F401
petsctools.cite("FiredrakeUserManual")
del petsctools

from ufl import *  # noqa: F401
from finat.ufl import *  # noqa: F401, isort: skip

from pyop2 import op2  # noqa: F401
from pyop2.mpi import COMM_SELF, COMM_WORLD  # noqa: F401

from firedrake.assemble import assemble  # noqa: F401
from firedrake.bcs import DirichletBC, EquationBC, homogenize  # noqa: F401
from firedrake.checkpointing import (  # noqa: F401
    FILE_CREATE,
    FILE_READ,
    FILE_UPDATE,
    CheckpointFile,
    DumbCheckpoint,
    HDF5File,
)
from firedrake.cofunction import Cofunction, RieszMap  # noqa: F401
from firedrake.constant import Constant  # noqa: F401
from firedrake.deflation import DeflatedSNES, Deflation  # noqa: F401
from firedrake.eigensolver import LinearEigenproblem, LinearEigensolver  # noqa: F401
from firedrake.ensemble import (  # noqa: F401
    Ensemble,
    EnsembleBJacobiPC,
    EnsembleBlockDiagonalMat,
    EnsembleCofunction,
    EnsembleDualSpace,
    EnsembleFunction,
    EnsembleFunctionSpace,
)
from firedrake.exceptions import (  # noqa: F401
    ConvergenceError,
    DofNotDefinedError,
    DofTypeError,
    FiredrakeException,
    MismatchingDomainError,
    VertexOnlyMeshMissingPointsError,
)
from firedrake.external_operators import (  # noqa: F401
    AbstractExternalOperator,
    MLOperator,
    PointexprOperator,
    assemble_method,
    point_expr,
)
from firedrake.function import (  # noqa: F401
    CoordinatelessFunction,
    Function,
    PointEvaluator,
    PointNotInDomainError,
)
from firedrake.functionspace import (  # noqa: F401
    FunctionSpace,
    MixedFunctionSpace,
    RestrictedFunctionSpace,
    TensorFunctionSpace,
    VectorFunctionSpace,
)
from firedrake.interpolation import (  # noqa: F401
    Interpolate,
    get_interpolator,
    interpolate,
)
from firedrake.linear_solver import LinearSolver  # noqa: F401
from firedrake.logging import (  # noqa: F401
    BLUE,
    CRITICAL,
    DEBUG,
    ERROR,
    GREEN,
    INFO,
    RED,
    WARNING,
    critical,
    debug,
    error,
    info,
    info_blue,
    info_green,
    info_red,
    log,
    set_level,
    set_log_handlers,
    set_log_level,
    warning,
)
from firedrake.matrix import (  # noqa: F401
    AssembledMatrix,
    ImplicitMatrix,
    Matrix,
    MatrixBase,
)
from firedrake.mesh import (  # noqa: F401
    DEFAULT_MESH_NAME,
    UNMARKED,
    AbstractMeshTopology,
    DistributedMeshOverlapType,
    ExtrudedMesh,
    ExtrudedMeshTopology,
    Mesh,
    MeshGeometry,
    MeshSequenceGeometry,
    MeshSequenceTopology,
    MeshTopology,
    RelabeledMesh,
    SubDomainData,
    Submesh,
    VertexOnlyMesh,
    VertexOnlyMeshTopology,
)
from firedrake.mg import (  # noqa: F401
    AdaptiveMeshHierarchy,
    AdaptiveTransferManager,
    ExtrudedMeshHierarchy,
    HierarchyBase,
    MeshHierarchy,
    NonNestedHierarchy,
    OpenCascadeMeshHierarchy,
    SemiCoarsenedExtrudedHierarchy,
    TransferManager,
    inject,
    prolong,
    restrict,
)
from firedrake.norms import errornorm, norm  # noqa: F401
from firedrake.nullspace import MixedVectorSpaceBasis, VectorSpaceBasis  # noqa: F401
from firedrake.output import VTKFile  # noqa: F401
from firedrake.parameters import (  # noqa: F401
    Parameters,
    disable_performance_optimisations,
    parameters,
)
from firedrake.parloops import (  # noqa: F401
    INC,
    MAX,
    MIN,
    READ,
    RW,
    WRITE,
    direct,
    par_loop,
)
from firedrake.petsc import PETSc  # noqa: F401
from firedrake.preconditioners import (  # noqa: F401
    BDDCPC,
    FDMPC,
    GTMGPC,
    LORPC,
    P1PC,
    P1SNES,
    PCDPC,
    PMGPC,
    PMGSNES,
    ASMExtrudedStarPC,
    ASMLinesmoothPC,
    ASMPatchPC,
    ASMStarPC,
    ASMVankaPC,
    AssembledPC,
    AuxiliaryOperatorPC,
    FacetSplitPC,
    HiptmairPC,
    HypreADS,
    HypreAMS,
    MassInvPC,
    PatchPC,
    PatchSNES,
    PCBase,
    PCSNESBase,
    PlaneSmoother,
    PoissonFDMPC,
    SNESBase,
    TwoLevelPC,
)
from firedrake.progress_bar import ProgressBar  # noqa: F401
from firedrake.projection import Projector, project  # noqa: F401
from firedrake.randomfunctiongen import *  # noqa: F401
from firedrake.slate import (  # noqa: F401
    SCPC,
    Add,
    AssembledVector,
    Block,
    BlockAssembledVector,
    DiagonalTensor,
    Factorization,
    HybridizationPC,
    Inverse,
    Mul,
    Negative,
    Reciprocal,
    SchurComplementBuilder,
    Solve,
    Tensor,
    TensorOp,
    Transpose,
)
from firedrake.slope_limiter import Limiter, VertexBasedLimiter  # noqa: F401
from firedrake.solving import solve  # noqa: F401
from firedrake.ufl_expr import (  # noqa: F401
    Argument,
    CellSize,
    Coargument,
    FacetNormal,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
    action,
    adjoint,
    derivative,
)
from firedrake.utility_meshes import (  # noqa: F401
    AnnulusMesh,
    BoxMesh,
    CircleManifoldMesh,
    CubedSphereMesh,
    CubeMesh,
    CylinderMesh,
    IcosahedralSphereMesh,
    IntervalMesh,
    OctahedralSphereMesh,
    PeriodicBoxMesh,
    PeriodicIntervalMesh,
    PeriodicRectangleMesh,
    PeriodicSquareMesh,
    PeriodicUnitCubeMesh,
    PeriodicUnitIntervalMesh,
    PeriodicUnitSquareMesh,
    RectangleMesh,
    SolidTorusMesh,
    SquareMesh,
    TensorBoxMesh,
    TensorRectangleMesh,
    TorusMesh,
    UnitBallMesh,
    UnitCubedSphereMesh,
    UnitCubeMesh,
    UnitDiskMesh,
    UnitIcosahedralSphereMesh,
    UnitIntervalMesh,
    UnitOctahedralSphereMesh,
    UnitSquareMesh,
    UnitTetrahedronMesh,
    UnitTriangleMesh,
)
from firedrake.variational_solver import (  # noqa: F401
    LinearVariationalProblem,
    LinearVariationalSolver,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
)

# Set default log level
set_log_level(WARNING)
set_log_handlers(comm=COMM_WORLD)

import sys

# Moved functionality
from firedrake._deprecation import plot  # noqa: F401

sys.modules["firedrake.plot"] = plot
from firedrake.plot import *  # noqa: F401

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
