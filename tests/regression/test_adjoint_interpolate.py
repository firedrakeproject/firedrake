import numpy as np
import pytest
from firedrake import *
from firedrake_adjoint import *
from pyadjoint.tape import get_working_tape, pause_annotation

def test_constant():
    mesh = UnitSquareMesh(10, 10)
    c = Constant(1.0, domain=mesh)
    V = FunctionSpace(mesh, "CG", 2)

    f = interpolate(c, V)
    J = assemble(f ** 2 * dx)
    Jhat = ReducedFunctional(J, Control(c))
    min_convergence_rate = taylor_test(Jhat, c, Constant(0.1, domain=mesh))

    # We need to clear the tape after each module
    tape = get_working_tape()
    tape.clear_tape()

    # Since we imported firedrake_adjoint, we need to pause annotation as well
    pause_annotation()

    assert np.isclose(min_convergence_rate, 2)