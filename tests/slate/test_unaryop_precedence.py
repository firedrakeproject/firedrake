from firedrake import *
import numpy
import pytest


@pytest.mark.skipif(utils.complex_mode, reason="Slate does not work for complex.")
def test_unary_minus():
    mesh = UnitSquareMesh(1, 1)

    V = FunctionSpace(mesh, "CG", 1)

    uh = Function(V)

    v = TestFunction(V)

    u = TrialFunction(V)

    A = Tensor(inner(u, v)*dx)

    B = Tensor(inner(uh, v)*dx)

    uh.assign(1)

    expr = action(A, uh) - B

    assert numpy.allclose(norm(assemble(expr)), 0)

    assert numpy.allclose(norm(assemble(-expr)), 0)
