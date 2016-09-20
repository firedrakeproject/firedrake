from __future__ import absolute_import
# Ensure petsc is initialised by us before anything else gets in there.
import firedrake.petsc as petsc
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
from pyop2 import op2                                           # noqa
from pyop2.mpi import COMM_WORLD, COMM_SELF                     # noqa

from firedrake.assemble import *
from firedrake.bcs import *
from firedrake.checkpointing import *
from firedrake.citations import *
from firedrake.constant import *
from firedrake.exceptions import *
from firedrake.expression import *
from firedrake.function import *
from firedrake.functionspace import *
from firedrake.interpolation import *
from firedrake.output import *
from firedrake.linear_solver import *
from firedrake.matrix_free.preconditioners import *
from firedrake.mesh import *
from firedrake.mg.mesh import *
from firedrake.mg.interface import *
from firedrake.norms import *
from firedrake.nullspace import *
from firedrake.optimizer import *
from firedrake.parameters import *
from firedrake.parloops import *
from firedrake.plot import *
from firedrake.projection import *
from firedrake.slope_limiter import *
from firedrake.solving import *
from firedrake.ufl_expr import *
from firedrake.utility_meshes import *
from firedrake.variational_solver import *
from firedrake.vector import *
from firedrake.version import __version__ as ver, __version_info__, check  # noqa

from firedrake.logging import *
# Set default log level
set_log_level(WARNING)
set_log_handlers(comm=COMM_WORLD)

check()
del check

from firedrake._version import get_versions
__version__ = get_versions()['version']
del get_versions
