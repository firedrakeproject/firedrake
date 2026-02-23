"""This demo program solves Poisson's equation

  - div grad u(x, y) = 0

on the unit square with boundary conditions given by:

  u(0, y) = 0
  v(1, y) = 42

Homogeneous Neumann boundary conditions are applied naturally on the
other two sides of the domain.

This has the analytical solution

  u(x, y) = 42*x[1]
"""
import pytest
from firedrake import *


def run_test(r, degree, parameters, quadrilateral=False):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    u = Function(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    bcs = [DirichletBC(V, Constant(0), 3),
           DirichletBC(V, Constant(42), 4)]

    # Compute solution
    solve(a == 0, u, solver_parameters=parameters, bcs=bcs)

    f = Function(V)
    f.interpolate(42*x[1])

    return sqrt(assemble(inner(u - f, u - f) * dx))


def run_test_linear(r, degree, parameters, quadrilateral=False):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(Constant(0), v) * dx

    bcs = [DirichletBC(V, Constant(0), 3),
           DirichletBC(V, Constant(42), 4)]

    # Compute solution
    u = Function(V)
    solve(a == L, u, solver_parameters=parameters, bcs=bcs)

    f = Function(V)
    f.interpolate(42*x[1])

    return sqrt(assemble(inner(u - f, u - f) * dx))


@pytest.mark.parametrize(['params', 'degree', 'quadrilateral'],
                         [(p, d, q)
                          for p in [{}, {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}]
                          for d in (1, 2)
                          for q in [False, True]])
def test_poisson_analytic(params, degree, quadrilateral):
    assert (run_test(2, degree, parameters=params, quadrilateral=quadrilateral) < 1.e-9)


@pytest.mark.parametrize(['params', 'degree', 'quadrilateral'],
                         [(p, d, q)
                          for p in [{}, {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}]
                          for d in (1, 2)
                          for q in [False, True]])
def test_poisson_analytic_linear(params, degree, quadrilateral):
    assert (run_test_linear(2, degree, parameters=params, quadrilateral=quadrilateral) < 5.e-6)


@pytest.mark.parallel(nprocs=2)
def test_poisson_analytic_linear_parallel():
    # specify superlu_dist as MUMPS fails in parallel on MacOS
    solver_parameters = {'pc_factor_mat_solver_type': 'superlu_dist'}
    error = run_test_linear(1, 1, solver_parameters)
    assert error < 5e-6
