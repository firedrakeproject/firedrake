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


def run_test(x, degree, parameters={}):
    # Create mesh and define function space
    mesh = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    u = Function(V)
    v = TestFunction(V)
    a = dot(grad(v), grad(u)) * dx

    bcs = [DirichletBC(V, 0, 1),
           DirichletBC(V, 42, 2)]

    # Compute solution
    solve(a == 0, u, solver_parameters=parameters, bcs=bcs)

    f = Function(V)
    f.interpolate(Expression("42*x[1]"))

    return sqrt(assemble(dot(u - f, u - f) * dx))


@pytest.mark.parametrize(['params', 'degree'],
                         [(p, d)
                          for p in [{}, {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}]
                          for d in (1, 2)])
def test_poisson_analytic(params, degree):
    assert (run_test(2, degree, parameters=params) < 1.e-9)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
