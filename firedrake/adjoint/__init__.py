"""The public interface to Firedrake's adjoint.

To start taping, run::

    from firedrake.adjoint import *
    continue_annotation()

"""
import pyadjoint
__version__ = pyadjoint.__version__

import sys
if 'backend' not in sys.modules:
    import firedrake
    sys.modules['backend'] = firedrake
else:
    raise ImportError("'backend' module already exists?")

from pyadjoint.tape import (Tape, set_working_tape, get_working_tape,
                            pause_annotation, continue_annotation,
                            stop_annotating, annotate_tape)
from pyadjoint.reduced_functional import ReducedFunctional
from firedrake.adjoint_utils.checkpointing import (
    enable_disk_checkpointing, pause_disk_checkpointing,
    continue_disk_checkpointing, stop_disk_checkpointing,
    checkpointable_mesh
)
from pyadjoint.verification import taylor_test, taylor_to_dict
from pyadjoint.drivers import compute_gradient, compute_hessian
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.control import Control
from pyadjoint import IPOPTSolver, ROLSolver, MinimizationProblem, InequalityConstraint, minimize

from firedrake.adjoint.ufl_constraints import UFLInequalityConstraint, UFLEqualityConstraint
import numpy_adjoint
import firedrake.ufl_expr
import types
import sys


# Work around the name clash of firedrake.adjoint vs ufl.adjoint.
# This will eventually become cleaner once we can rely on users having 
# Python 3.12 (see PEP 713).
class AdjointModule(types.ModuleType):
    def __call__(self, form):
        return firedrake.ufl_expr.adjoint(form)


sys.modules[__name__].__class__ = AdjointModule

set_working_tape(Tape())
