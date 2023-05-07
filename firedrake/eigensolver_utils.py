from itertools import chain

import numpy

from pyop2 import op2
from firedrake_configuration import get_config
from firedrake import function, dmhooks
from firedrake.exceptions import ConvergenceError
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.utils import cached_property
from firedrake.logging import warning
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)

def _make_reasons(reasons):
    return dict([(getattr(reasons, r), r)
                 for r in dir(reasons) if not r.startswith('_')])

EPSReasons = _make_reasons(SLEPc.EPS.ConvergedReason())


def check_eps_convergence(eps):
    r = eps.getConvergedReason()
    try:
        reason = SLEPc.EPS.ConvergedReasons[r]
    except KeyError:
        reason = "unknown reason (petsc4py enum incomplete?), try with -eps_converged_reason and -nep_converged_reason"
    if r < 0:
        raise ConvergenceError(r"""Eigenproblem failed to converge after %d iterations.
Reason:
   %s""" % (eps.getIterationNumber(), reason))