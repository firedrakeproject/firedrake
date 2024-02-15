"""Solves the Helmholtz equation with Lagrange and Bernstein
polynomials, and checks that the respective solutions are consistent
with each other.  This is to ensure that Bernstein elements function
correctly."""

from os.path import abspath, dirname
import numpy as np
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


@pytest.fixture(params=[1, 2, 3])
def mesh(request):
    dim = request.param
    if dim == 1:
        return UnitIntervalMesh(10)
    elif dim == 2:
        return UnitSquareMesh(8, 8)
    elif dim == 3:
        return UnitCubeMesh(4, 4, 4)


@pytest.mark.parametrize('degree', [1, 2, 3, 4])
def test_bernstein(mesh, degree):
    # Solve with Bernstein polynomials
    B = FunctionSpace(mesh, "Bernstein", degree)
    xb = helmholtz(B)

    # Solve with Lagrange polynomials
    L = FunctionSpace(mesh, "Lagrange", degree)
    xl = helmholtz(L)

    # Convert Bernstein solution to Lagrange space, and compare it
    # with Lagrange solution
    xp = Function(L).interpolate(xb)
    assert np.allclose(xl.dat.data, xp.dat.data)


def helmholtz(V):
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x = SpatialCoordinate(V.mesh())
    f.project(np.prod([cos(2*pi*xi) for xi in x]))
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx

    # Compute solution
    x = Function(V)
    solve(a == L, x, solver_parameters={'ksp_type': 'cg', 'pc_type': 'lu'})
    return x
