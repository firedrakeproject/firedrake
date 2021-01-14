import numpy as np

from firedrake import *


def test_jac_invalid():
    mesh = UnitIntervalMesh(5)
    V = FunctionSpace(mesh, "P", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V).assign(1)

    # set up trivial projection problem
    a = inner(u, v)*dx
    L = inner(f, v)*dx

    out = Function(V)
    problem = LinearVariationalProblem(a, L, out, constant_jacobian=True)
    solver = LinearVariationalSolver(problem)
    solver.solve()

    assert np.allclose(out.dat.data, 1.0)
    # re-zero output vector, else action(a, out) - L,
    # the residual, is still zero
    out.assign(0)

    # change mesh without invalidating assembled matrix
    mesh.coordinates.dat.data[:] *= 2.0
    solver.solve()

    # wrong answer will be produced
    assert not np.allclose(out.dat.data, 1.0)
    out.assign(0)

    # now invalidate the assembled matrix, forcing reassembly on modified mesh
    solver.invalidate_jacobian()
    solver.solve()

    # correct answer produced
    assert np.allclose(out.dat.data, 1.0)
