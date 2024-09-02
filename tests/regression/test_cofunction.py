import pytest
import numpy as np
from firedrake import *


@pytest.fixture
def V():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    return V


def test_cofunction_assign_cofunction_with_subset(V):
    f = Cofunction(V.dual())
    subset = op2.Subset(V.node_set, [0, 1, 2])
    f.dat.data[:] = 1.0
    assert np.allclose(f.dat.data_ro, 1.0)

    g = Cofunction(V.dual())
    g.dat.data[:] = 2.0

    f.assign(g, subset=subset)
    assert np.allclose(f.dat.data_ro[:3], 2.0)
    assert np.allclose(f.dat.data_ro[3:], 1.0)


def test_cofunction_assign_scaled_cofunction_with_subset(V):
    f = Cofunction(V.dual())
    subset = op2.Subset(V.node_set, [0, 1, 2])
    f.dat.data[:] = 1.0
    assert np.allclose(f.dat.data_ro, 1.0)

    g = Cofunction(V.dual())
    g.dat.data[:] = 2.0

    f.assign(-3 * g, subset=subset)
    assert np.allclose(f.dat.data_ro[:3], -6.0)
    assert np.allclose(f.dat.data_ro[3:], 1.0)


def test_scalar_cofunction_zero(V):
    f = Cofunction(V.dual())

    f.dat.data[:] = 1

    g = f.zero()
    assert f is g
    assert np.allclose(f.dat.data_ro, 0.0)


def test_scalar_cofunction_zero_with_subset(V):
    f = Cofunction(V.dual())
    # create an arbitrary subset consisting of the first two nodes
    assert V.node_set.size > 2
    subset = op2.Subset(V.node_set, [0, 1])

    f.dat.data[:] = 1

    g = f.zero(subset=subset)
    assert f is g
    assert np.allclose(f.dat.data_ro[:2], 0.0)
    assert np.allclose(f.dat.data_ro[2:], 1.0)


def test_diriclet_bc_rhs(V):
    # Issue https://github.com/firedrakeproject/firedrake/issues/3498
    # Apply DirichletBC to RHS (Cofunction) in LinearVariationalSolver
    mesh = UnitIntervalMesh(2)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    # Form RHS
    u = Function(space, name="u")
    problem = LinearVariationalProblem(
        inner(trial, test) * dx, inner(Constant(1.0), test) * dx, u,
        DirichletBC(space, 0.0, "on_boundary"))
    solver = LinearVariationalSolver(problem)
    solver.solve()

    assert np.allclose(assemble(inner(u, u) * ds), 0.0)

    # Cofunction RHS
    b = assemble(inner(Constant(1.0), test) * dx)
    u = Function(space, name="u")
    problem = LinearVariationalProblem(
        inner(trial, test) * dx, b, u,
        DirichletBC(space, 0.0, "on_boundary"))
    solver = LinearVariationalSolver(problem)
    solver.solve()

    assert np.allclose(assemble(inner(u, u) * ds), 0.0)

    # FormSum RHS
    b = assemble(inner(Constant(0.5), test) * dx) + inner(Constant(0.5), test) * dx
    u = Function(space, name="u")
    problem = LinearVariationalProblem(
        inner(trial, test) * dx, b, u,
        DirichletBC(space, 0.0, "on_boundary"))
    solver = LinearVariationalSolver(problem)
    solver.solve()

    assert np.allclose(assemble(inner(u, u) * ds), 0.0)
