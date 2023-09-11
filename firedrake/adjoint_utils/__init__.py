"""Infrastructure for Firedrake's adjoint.

This subpackage contains the Firedrake-specific code required to interface with
:mod:`pyadjoint`. For the public interface to Firedrake's adjoint, please see
:mod:`firedrake.adjoint`.

"""
from firedrake.adjoint_utils.function import *               # noqa: F401
from firedrake.adjoint_utils.assembly import *               # noqa: F401
from firedrake.adjoint_utils.projection import *             # noqa: F401
from firedrake.adjoint_utils.variational_solver import *     # noqa: F401
from firedrake.adjoint_utils.solving import *                # noqa: F401
from firedrake.adjoint_utils.mesh import *                   # noqa: F401
from firedrake.adjoint_utils.checkpointing import *          # noqa: F401
