"""Infrastructure for Firedrake's adjoint.

This subpackage contains the Firedrake-specific code required to interface with
:mod:`pyadjoint`. For the public interface to Firedrake's adjoint, please see
:mod:`firedrake.adjoint`.

"""
from firedrake.adjoint_utils.function import (  # noqa F401
    FunctionMixin, CofunctionMixin
)
from firedrake.adjoint_utils.assembly import annotate_assemble  # noqa F401
from firedrake.adjoint_utils.projection import annotate_project  # noqa F401
from firedrake.adjoint_utils.variational_solver import (  # noqa F401
    NonlinearVariationalProblemMixin, NonlinearVariationalSolverMixin
)
from firedrake.adjoint_utils.solving import annotate_solve, get_solve_blocks  # noqa F401
from firedrake.adjoint_utils.mesh import MeshGeometryMixin  # noqa F401
from firedrake.adjoint_utils.checkpointing import (  # noqa F401
    enable_disk_checkpointing, disk_checkpointing,
    pause_disk_checkpointing, continue_disk_checkpointing,
    stop_disk_checkpointing, checkpointable_mesh
)
from firedrake.adjoint_utils.ensemble_function import EnsembleFunctionMixin  # noqa F401
