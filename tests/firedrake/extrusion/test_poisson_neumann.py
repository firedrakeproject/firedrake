"""This demo program solves Poisson's equation

  - div grad u(x, y, z) = f

in a unit cube, with Dirichlet boundary conditions on 4 sides
and Neumann boundary conditions (one explicit, one implicit)
on the other 2, opposite, sides.
"""

import pytest
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def P2(extmesh, request):
    quadrilateral = request.param
    mesh = extmesh(4, 4, 4, quadrilateral=quadrilateral)
    return FunctionSpace(mesh, "CG", 2)


def test_bottom(P2):
    u = TrialFunction(P2)
    v = TestFunction(P2)
    xs = SpatialCoordinate(P2.mesh())

    a = inner(grad(u), grad(v))*dx
    L = -inner(20, v)*dx + inner(20, v)*ds_b
    bc_expr = 10*(xs[2]-1)*(xs[2]-1)
    bcs = [DirichletBC(P2, bc_expr, 1),
           DirichletBC(P2, bc_expr, 2),
           DirichletBC(P2, bc_expr, 3),
           DirichletBC(P2, bc_expr, 4)]

    u = Function(P2)
    solve(a == L, u, bcs)

    u_exact = Function(P2)
    u_exact.interpolate(bc_expr)

    assert max(abs(u.dat.data - u_exact.dat.data)) < 1.0e-6


def test_top(P2):
    u = TrialFunction(P2)
    v = TestFunction(P2)
    xs = SpatialCoordinate(P2.mesh())

    a = inner(grad(u), grad(v))*dx
    L = -inner(20, v)*dx + inner(20, v)*ds_t
    bc_expr = 10*xs[2]*xs[2]
    bcs = [DirichletBC(P2, bc_expr, 1),
           DirichletBC(P2, bc_expr, 2),
           DirichletBC(P2, bc_expr, 3),
           DirichletBC(P2, bc_expr, 4)]

    u = Function(P2)
    solve(a == L, u, bcs)

    u_exact = Function(P2)
    u_exact.interpolate(bc_expr)

    assert max(abs(u.dat.data - u_exact.dat.data)) < 1.0e-6


def test_topbottom(P2):
    u = TrialFunction(P2)
    v = TestFunction(P2)
    xs = SpatialCoordinate(P2.mesh())

    a = inner(grad(u), grad(v))*dx
    L = -inner(20, v)*dx + inner(10, v)*ds_tb
    bc_expr = 10*(xs[2]-0.5)*(xs[2]-0.5)
    bcs = [DirichletBC(P2, bc_expr, 1),
           DirichletBC(P2, bc_expr, 2),
           DirichletBC(P2, bc_expr, 3),
           DirichletBC(P2, bc_expr, 4)]

    u = Function(P2)
    solve(a == L, u, bcs)

    u_exact = Function(P2)
    u_exact.interpolate(bc_expr)

    assert max(abs(u.dat.data - u_exact.dat.data)) < 1.0e-6
