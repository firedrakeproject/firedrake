
"""The public interface to Firedrake's adjoint.

To start taping, run::

    from firedrake.adjoint import *
    continue_annotation()

"""
import pyadjoint

__version__ = pyadjoint.__version__

import sys
import types

import numpy_adjoint  # noqa F401

from pyadjoint import (  # noqa: F401
    InequalityConstraint,
    IPOPTSolver,
    MinimizationProblem,
    ROLSolver,
    minimize,
)
from pyadjoint.adjfloat import AdjFloat  # noqa F401
from pyadjoint.checkpointing import disk_checkpointing_callback  # noqa F401
from pyadjoint.control import Control  # noqa F401
from pyadjoint.drivers import (  # noqa F401
    compute_derivative,
    compute_gradient,
    compute_hessian,
)
from pyadjoint.reduced_functional import ReducedFunctional  # noqa F401
from pyadjoint.tape import (  # noqa: F401
    Tape,
    annotate_tape,
    continue_annotation,
    get_working_tape,
    pause_annotation,
    set_working_tape,
    stop_annotating,
)
from pyadjoint.verification import taylor_test, taylor_to_dict  # noqa F401

import firedrake.ufl_expr
from firedrake.adjoint.covariance_operator import (  # noqa F401
    AutoregressiveCovariance,
    CovarianceMat,
    CovariancePC,
    PetscNoiseBackend,
    PyOP2NoiseBackend,
    VOMNoiseBackend,
    WhiteNoiseGenerator,
)
from firedrake.adjoint.ensemble_reduced_functional import (  # noqa F401
    EnsembleReducedFunctional,
)
from firedrake.adjoint.transformed_functional import (  # noqa: F401
    L2RieszMap,
    L2TransformedFunctional,
)
from firedrake.adjoint.ufl_constraints import (  # noqa: F401
    UFLEqualityConstraint,
    UFLInequalityConstraint,
)
from firedrake.adjoint_utils.checkpointing import (  # noqa: F401
    checkpointable_mesh,
    continue_disk_checkpointing,
    enable_disk_checkpointing,
    pause_disk_checkpointing,
    stop_disk_checkpointing,
)
from firedrake.adjoint_utils.solving import get_solve_blocks  # noqa F401

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
