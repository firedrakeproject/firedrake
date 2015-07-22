# Must happen first, to ensure prefork server is up and running.
from pyop2 import op2                                           # noqa
# Ensure petsc is initialised by us before anything else gets in there.
import petsc
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
from pyop2.logger import set_log_level, info_red, info_green, info_blue, log  # noqa
from pyop2.logger import debug, info, warning, error, critical  # noqa
from pyop2.logger import DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa

from assemble import *
from bcs import *
from constant import *
from expression import *
from function import *
from functionspace import *
from io import *
from linear_solver import *
from mesh import *
from mg import *
from norms import *
from nullspace import *
from optimizer import *
from parameters import *
from parloops import *
from projection import *
from solving import *
from ufl_expr import *
from utility_meshes import *
from variational_solver import *
from vector import *
from version import __version__ as ver, __version_info__, check  # noqa

# Set default log level
set_log_level(INFO)

check()
del check

from ._version import get_versions
__version__ = get_versions(default={"version": ver, "full": ""})['version']
del get_versions
