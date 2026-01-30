
"""The public interface to Firedrake's adjoint.

To start taping, run::

    from firedrake.adjoint import *
    continue_annotation()

"""
import pyadjoint
__version__ = pyadjoint.__version__

from pyadjoint.tape import (  # noqa: F401
    Tape, set_working_tape, get_working_tape, pause_annotation,
    continue_annotation, stop_annotating, annotate_tape
)
from pyadjoint.reduced_functional import ReducedFunctional  # noqa F401
from pyadjoint.checkpointing import disk_checkpointing_callback  # noqa F401
from firedrake.adjoint_utils.checkpointing import (  # noqa: F401
    enable_disk_checkpointing, pause_disk_checkpointing,
    continue_disk_checkpointing, stop_disk_checkpointing,
    checkpointable_mesh
)
from firedrake.adjoint_utils import get_solve_blocks  # noqa F401
from pyadjoint.verification import taylor_test, taylor_to_dict  # noqa F401
from pyadjoint.drivers import compute_gradient, compute_derivative, compute_hessian  # noqa F401
from pyadjoint.adjfloat import AdjFloat  # noqa F401
from pyadjoint.control import Control  # noqa F401
from pyadjoint import (  # noqa: F401
    IPOPTSolver, ROLSolver, MinimizationProblem, InequalityConstraint, minimize
)
from firedrake.adjoint.ufl_constraints import (  # noqa: F401
    UFLInequalityConstraint, UFLEqualityConstraint
)
from firedrake.adjoint.ensemble_reduced_functional import EnsembleReducedFunctional  # noqa F401
from firedrake.adjoint.transformed_functional import L2RieszMap, L2TransformedFunctional  # noqa: F401
from firedrake.adjoint.covariance_operator import (  # noqa F401
    WhiteNoiseGenerator, AutoregressiveCovariance,
    PyOP2NoiseBackend, PetscNoiseBackend, VOMNoiseBackend,
    CovarianceMat, CovariancePC)
import numpy_adjoint  # noqa F401
import firedrake.ufl_expr
import types
import sys

# Work around the name clash of firedrake.adjoint vs ufl.adjoint.
# This will eventually become cleaner once we can rely on users having
# Python 3.12 (see PEP 713).


class _AdjointModule(types.ModuleType):
    def __call__(self, form):
        return firedrake.ufl_expr.adjoint(form)


sys.modules[__name__].__class__ = _AdjointModule
del sys
del types

set_working_tape(Tape())
