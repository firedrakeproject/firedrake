from firedrake import *
import numpy as np


def test_empty_integrand():
    m = UnitSquareMesh(5, 5)
    mesh = ExtrudedMesh(m, layers=5)

    P1 = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(P1)
    v = TestFunction(P1)
    sol = Function(P1)
    f = Function(P1)
    f.interpolate(Constant(4.5))

    A = inner(u, v)*dx
    L = inner(f, v)*dx + inner(f, v)*ds_v

    u = Function(P1)
    solve(A == L, u)

    F = inner(sol, v)*dx - inner(f, v)*dx - inner(f, v)*ds_v
    solve(F == 0, sol)

    assert np.allclose(u.dat.data_ro, sol.dat.data_ro)
