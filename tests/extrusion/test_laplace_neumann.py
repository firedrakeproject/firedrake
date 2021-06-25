"""This demo program solves Laplace's equation

  - div grad u(x, y, z) = 0

in a unit square or unit cube, with Dirichlet boundary
conditions on 2/4 sides and Neumann boundary conditions
on the other 2, opposite, sides.
"""

import pytest
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def P2(extmesh, request):
    quadrilateral = request.param
    mesh = extmesh(4, 4, 4, quadrilateral=quadrilateral)
    return FunctionSpace(mesh, "CG", 2)


@pytest.fixture(scope='module')
def P2_2D(extmesh_2D):
    mesh = extmesh_2D(4, 4)
    return FunctionSpace(mesh, "CG", 2)


def test_bottom_and_top(P2):
    u = TrialFunction(P2)
    v = TestFunction(P2)

    a = inner(grad(u), grad(v))*dx
    L = inner(10, v)*ds_b - inner(10, v)*ds_t
    xs = SpatialCoordinate(P2.mesh())
    bc_expr = -10*xs[2]
    bcs = [DirichletBC(P2, bc_expr, 1),
           DirichletBC(P2, bc_expr, 2),
           DirichletBC(P2, bc_expr, 3),
           DirichletBC(P2, bc_expr, 4)]

    u = Function(P2)
    solve(a == L, u, bcs)

    u_exact = Function(P2)
    u_exact.interpolate(bc_expr)

    assert max(abs(u.dat.data - u_exact.dat.data)) < 1.1e-6


def test_top_and_bottom(P2):
    u = TrialFunction(P2)
    v = TestFunction(P2)

    a = inner(grad(u), grad(v))*dx
    L = inner(10, v)*ds_t - inner(10, v)*ds_b
    xs = SpatialCoordinate(P2.mesh())
    bc_expr = 10*xs[2]
    bcs = [DirichletBC(P2, bc_expr, 1),
           DirichletBC(P2, bc_expr, 2),
           DirichletBC(P2, bc_expr, 3),
           DirichletBC(P2, bc_expr, 4)]

    u = Function(P2)
    solve(a == L, u, bcs)

    u_exact = Function(P2)
    u_exact.interpolate(bc_expr)

    assert max(abs(u.dat.data - u_exact.dat.data)) < 1.1e-6


def test_left_right(P2):
    u = TrialFunction(P2)
    v = TestFunction(P2)

    a = inner(grad(u), grad(v))*dx
    L = inner(10, v)*ds_v(2) - inner(10, v)*ds_v(1)
    xs = SpatialCoordinate(P2.mesh())
    bc_expr = 10*xs[0]
    bcs = [DirichletBC(P2, bc_expr, "top"),
           DirichletBC(P2, bc_expr, "bottom"),
           DirichletBC(P2, bc_expr, 3),
           DirichletBC(P2, bc_expr, 4)]

    u = Function(P2)
    solve(a == L, u, bcs)

    u_exact = Function(P2)
    u_exact.interpolate(bc_expr)

    assert max(abs(u.dat.data - u_exact.dat.data)) < 1.1e-6


def test_near_far(P2):
    u = TrialFunction(P2)
    v = TestFunction(P2)

    xs = SpatialCoordinate(P2.mesh())
    a = inner(grad(u), grad(v))*dx
    L = inner(10, v)*ds_v(4) - inner(10, v)*ds_v(3)
    bc_expr = 10*xs[1]
    bcs = [DirichletBC(P2, bc_expr, 1),
           DirichletBC(P2, bc_expr, 2),
           DirichletBC(P2, bc_expr, "top"),
           DirichletBC(P2, bc_expr, "bottom")]

    u = Function(P2)
    solve(a == L, u, bcs)

    u_exact = Function(P2)
    u_exact.interpolate(bc_expr)

    assert max(abs(u.dat.data - u_exact.dat.data)) < 1.0e-6


def test_2D_bottom_top(P2_2D):
    u = TrialFunction(P2_2D)
    v = TestFunction(P2_2D)

    xs = SpatialCoordinate(P2_2D.mesh())
    a = inner(grad(u), grad(v))*dx
    L = inner(10, v)*ds_t - inner(10, v)*ds_b
    bc_expr = 10*xs[1]
    bcs = [DirichletBC(P2_2D, bc_expr, 1),
           DirichletBC(P2_2D, bc_expr, 2)]

    u = Function(P2_2D)
    solve(a == L, u, bcs)

    u_exact = Function(P2_2D)
    u_exact.interpolate(bc_expr)

    assert max(abs(u.dat.data - u_exact.dat.data)) < 1.0e-6


def test_2D_left_right(P2_2D):
    u = TrialFunction(P2_2D)
    v = TestFunction(P2_2D)

    xs = SpatialCoordinate(P2_2D.mesh())
    a = inner(grad(u), grad(v))*dx
    L = inner(10, v)*ds_v(2) - inner(10, v)*ds_v(1)
    bc_expr = 10*xs[0]
    bcs = [DirichletBC(P2_2D, bc_expr, "top"),
           DirichletBC(P2_2D, bc_expr, "bottom")]

    u = Function(P2_2D)
    solve(a == L, u, bcs)

    u_exact = Function(P2_2D)
    u_exact.interpolate(bc_expr)

    assert max(abs(u.dat.data - u_exact.dat.data)) < 1.0e-6
