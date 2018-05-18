from firedrake import *
import numpy


def test_unary_minus():
    mesh = UnitSquareMesh(1, 1)

    V = FunctionSpace(mesh, "CG", 1)

    uh = Function(V)

    v = TestFunction(V)

    u = TrialFunction(V)

    A = Tensor(u*v*dx)

    B = Tensor(uh*v*dx)

    uh.assign(1)

    expr = action(A, uh) - B

    assert numpy.allclose(norm(assemble(expr)), 0)

    assert numpy.allclose(norm(assemble(-expr)), 0)
