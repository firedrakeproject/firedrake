"""Infrastructure for Firedrake's adjoint.

This subpackage contains the Firedrake-specific code required to interface with
:mod:`pyadjoint`. For the public interface to Firedrake's adjoint, please see
:mod:`firedrake.adjoint`.

"""
from firedrake.adjoint_utils.assembly import annotate_assemble  # noqa F401
from firedrake.adjoint_utils.checkpointing import (  # noqa F401
    checkpointable_mesh,
    continue_disk_checkpointing,
    disk_checkpointing,
    enable_disk_checkpointing,
    pause_disk_checkpointing,
    stop_disk_checkpointing,
)
from firedrake.adjoint_utils.ensemble_function import EnsembleFunctionMixin  # noqa F401
from firedrake.adjoint_utils.function import CofunctionMixin, FunctionMixin  # noqa F401
from firedrake.adjoint_utils.mesh import MeshGeometryMixin  # noqa F401
from firedrake.adjoint_utils.projection import annotate_project  # noqa F401
from firedrake.adjoint_utils.solving import (  # noqa F401
    annotate_solve,
    get_solve_blocks,
)
from firedrake.adjoint_utils.variational_solver import (  # noqa F401
    NonlinearVariationalProblemMixin,
    NonlinearVariationalSolverMixin,
)
