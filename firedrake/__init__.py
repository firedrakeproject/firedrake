import functools

# The range of PETSc versions supported by Firedrake. Note that unlike in
# firedrake-configure and pyproject.toml where we want to be strict about
# the specific version, here we are more permissive. This is to catch the
# case where users don't update their PETSc for a really long time or
# accidentally install a too-new release that isn't yet supported.
PETSC_SUPPORTED_VERSIONS = ">=3.25.0"


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
    from firedrake.logging import info

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


def warn_omp_num_threads():
    import os
    from firedrake.logging import warning

    if os.getenv("OMP_NUM_THREADS") != "1":
        warning("OMP_NUM_THREADS is not set or is set to a value greater than 1, "
                "we suggest setting OMP_NUM_THREADS=1 to improve performance")


_initializing = False


# Runs once; a raising call is not cached, so a failed init is retryable.
@functools.cache
def initialize():
    global _initializing
    _initializing = True
    try:
        _run_init()
    finally:
        _initializing = False


def _run_init():
    import importlib

    def star(modname):  # from <modname> import * (illegal inside a function)
        mod = importlib.import_module(modname)
        names = getattr(mod, "__all__", None) or [n for n in dir(mod) if not n.startswith("_")]
        globals().update({n: getattr(mod, n) for n in names})

    # Ensure petsc is initialised right away
    init_petsc()

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

    star("ufl")
    star("finat.ufl")

    from pyop2 import op2                        # noqa: F401
    from pyop2.mpi import COMM_WORLD, COMM_SELF  # noqa: F401

    # Register possible citations
    import firedrake.citations  # noqa: F401
    petsctools.cite("FiredrakeUserManual")
    del petsctools

    from firedrake.petsc import PETSc  # noqa: F401
    from firedrake.assemble import assemble  # noqa: F401
    from firedrake.bcs import DirichletBC, homogenize, EquationBC  # noqa: F401
    from firedrake.checkpointing import (  # noqa: F401
        DumbCheckpoint, HDF5File, FILE_READ, FILE_CREATE,
        FILE_UPDATE, CheckpointFile
    )
    from firedrake.cofunction import Cofunction, RieszMap  # noqa: F401
    from firedrake.constant import Constant  # noqa: F401
    from firedrake.deflation import DeflatedSNES, Deflation  # noqa: F401
    from firedrake.exceptions import (  # noqa: F401
        FiredrakeException, ConvergenceError, MismatchingDomainError,
        VertexOnlyMeshMissingPointsError, DofNotDefinedError, DofTypeError,
        SerialExecutionOnlyError, PointNotInDomainError,
    )
    from firedrake.function import (  # noqa: F401
        Function, CoordinatelessFunction, PointEvaluator
    )
    from firedrake.functionspace import (  # noqa: F401
        MixedFunctionSpace, FunctionSpace, VectorFunctionSpace,
        TensorFunctionSpace, RestrictedFunctionSpace
    )
    from firedrake.interpolation import (  # noqa: F401
        interpolate, Interpolate, get_interpolator
    )
    from firedrake.linear_solver import LinearSolver  # noqa: F401
    from firedrake.preconditioners import (  # noqa: F401
        PCBase, SNESBase, PCSNESBase, ASMPatchPC, ASMStarPC, ASMVankaPC,
        ASMLinesmoothPC, ASMExtrudedStarPC, AssembledPC, AuxiliaryOperatorPC,
        MassInvPC, PCDPC, PatchPC, PlaneSmoother, PatchSNES, P1PC, P1SNES,
        LORPC, GTMGPC, PMGPC, PMGSNES, HypreAMS, HypreADS, FDMPC,
        PoissonFDMPC, TwoLevelPC, HiptmairPC, FacetSplitPC, BDDCPC,
        CovariancePC, OffloadPC
    )
    from firedrake.mesh import (  # noqa: F401
        Mesh, ExtrudedMesh, VertexOnlyMesh, RelabeledMesh,
        SubDomainData, UNMARKED, DistributedMeshOverlapType,
        DEFAULT_MESH_NAME, MeshGeometry, MeshTopology,
        AbstractMeshTopology, ExtrudedMeshTopology, Submesh,
        VertexOnlyMeshTopology, MeshSequenceGeometry, MeshSequenceTopology
    )
    from firedrake.mg import (  # noqa: F401
        HierarchyBase, MeshHierarchy, ExtrudedMeshHierarchy,
        NonNestedHierarchy, SemiCoarsenedExtrudedHierarchy, SubmeshHierarchy,
        prolong, restrict, inject, TransferManager,
        OpenCascadeMeshHierarchy, AdaptiveMeshHierarchy,
        AdaptiveTransferManager
    )
    from firedrake.norms import errornorm, norm  # noqa: F401
    from firedrake.nullspace import VectorSpaceBasis, MixedVectorSpaceBasis  # noqa: F401
    from firedrake.output import VTKFile  # noqa: F401
    from firedrake.parameters import (  # noqa: F401
        Parameters, parameters, disable_performance_optimisations
    )
    from firedrake.parloops import (  # noqa: F401
        par_loop, direct, READ, WRITE, RW, INC, MIN, MAX
    )
    from firedrake.projection import (  # noqa: F401
        project, Projector
    )
    from firedrake.slate import (  # noqa: F401
        AssembledVector, Block, Factorization, Tensor, Inverse,
        Transpose, Negative, Add, Mul, Solve, BlockAssembledVector,
        DiagonalTensor, Reciprocal, HybridizationPC, SchurComplementBuilder,
        SCPC, TensorOp
    )
    from firedrake.slope_limiter import (  # noqa: F401
        Limiter, VertexBasedLimiter
    )
    from firedrake.solving import solve  # noqa: F401
    from firedrake.ufl_expr import (  # noqa: F401
        Argument, Coargument, TestFunction, TrialFunction,
        TestFunctions, TrialFunctions, derivative, adjoint,
        action, CellSize, FacetNormal
    )
    from firedrake.utility_meshes import (  # noqa: F401
        IntervalMesh, UnitIntervalMesh, PeriodicIntervalMesh,
        PeriodicUnitIntervalMesh, UnitTriangleMesh, RectangleMesh,
        TensorRectangleMesh, SquareMesh, UnitSquareMesh, PeriodicRectangleMesh,
        PeriodicSquareMesh, PeriodicUnitSquareMesh, CircleManifoldMesh,
        UnitDiskMesh, UnitBallMesh, UnitTetrahedronMesh, TensorBoxMesh,
        BoxMesh, CubeMesh, UnitCubeMesh, PeriodicBoxMesh, PeriodicUnitCubeMesh,
        IcosahedralSphereMesh, UnitIcosahedralSphereMesh, OctahedralSphereMesh,
        UnitOctahedralSphereMesh, CubedSphereMesh, UnitCubedSphereMesh,
        TorusMesh, AnnulusMesh, SolidTorusMesh, CylinderMesh
    )
    from firedrake.variational_solver import (  # noqa: F401
        LinearVariationalProblem, LinearVariationalSolver,
        NonlinearVariationalProblem, NonlinearVariationalSolver
    )
    from firedrake.eigensolver import (  # noqa: F401
        LinearEigenproblem, LinearEigensolver
    )
    from firedrake.ensemble import (  # noqa: F401
        Ensemble, EnsembleFunction, EnsembleCofunction,
        EnsembleFunctionSpace, EnsembleDualSpace, EnsembleBJacobiPC,
        EnsembleBlockDiagonalMat
    )
    star("firedrake.randomfunctiongen")
    from firedrake.external_operators import (  # noqa: F401
        AbstractExternalOperator, assemble_method,
        PointexprOperator, point_expr, MLOperator
    )
    from firedrake.progress_bar import ProgressBar  # noqa: F401

    from firedrake.logging import (  # noqa: F401
        set_level, set_log_handlers, set_log_level, DEBUG, INFO,
        WARNING, ERROR, CRITICAL, log, debug, info, warning, error,
        critical, info_red, info_green, info_blue, RED, GREEN, BLUE
    )
    from firedrake.matrix import (  # noqa: F401
        MatrixBase, Matrix, ImplicitMatrix, AssembledMatrix
    )

    # Set default log level
    set_log_level(WARNING)
    set_log_handlers(comm=COMM_WORLD)

    # Moved functionality
    import sys
    from firedrake._deprecation import plot  # noqa: F401
    sys.modules["firedrake.plot"] = plot
    star("firedrake.plot")

    set_blas_num_threads()
    warn_omp_num_threads()

    globals().update({k: v for k, v in locals().items() if not k.startswith("_") and k != "star"})

    # Stop profiling Firedrake import
    if _is_logging:
        _init_event.end()


def __getattr__(name):
    # A submodule imported during _run_init may do `from firedrake import X`
    # before X is bound; don't re-enter init, let normal import resolution run.
    if _initializing:
        raise AttributeError(name)
    initialize()
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
