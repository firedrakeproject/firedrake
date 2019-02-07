from firedrake import *
import numpy as np


def test_empty_integrand():
    mesh = UnitSquareMesh(5, 5)

    P1 = FiniteElement("CG", triangle, 1)
    B = FiniteElement("B", triangle, 3)
    Mini = FunctionSpace(mesh, P1+B)

    u = TrialFunction(Mini)
    v = TestFunction(Mini)
    sol = Function(Mini)
    f = Function(Mini)
    f.assign(1)

    a = inner(u, v)*dx
    L = inner(f, v)*dx + inner(f, v)*ds(3)

    u = Function(Mini)
    solve(a == L, u)

    # At one point, this test would have failed since we use an EnrichedElement
    # rather than a FiniteElement or VectorElement.
    # Note the failure mode is an error during the solve, not a failed assertion.
    F = inner(sol, v)*dx - inner(f, v)*dx - inner(f, v)*ds(3)
    solve(F == 0, sol)

    assert np.allclose(u.dat.data_ro, sol.dat.data_ro)
