from firedrake.adjoint.function import *               # noqa: F401
from firedrake.adjoint.assembly import *               # noqa: F401
from firedrake.adjoint.projection import *             # noqa: F401
from firedrake.adjoint.variational_solver import *     # noqa: F401
from firedrake.adjoint.solving import *                # noqa: F401
from firedrake.adjoint.mesh import *                   # noqa: F401
from firedrake.adjoint.interpolate import *            # noqa: F401
from pyadjoint.tape import Tape, set_working_tape

set_working_tape(Tape())
