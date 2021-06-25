"""Solve the H^1 Helmholtz problem (with lambda = 1) on a
5x3 periodic rectangular mesh.  We use the exact solution

    u = sin(2*pi*(2*x[0]/5))*sin(2*pi*(x[1]/3))

This requires f = [(244/225)*pi^2 + 1]u.

This test will fail on a non-periodic mesh since the integration
by parts generates a term involving the normal derivative of
u on the boundary, unless Dirichlet BCs are set.

On a doubly-periodic mesh, no boundary conditions are required.  On a
partially periodic mesh, we impose zero Dirichlet BCs on the
non-periodic boundaries.
"""

import pytest
from math import pi

from firedrake import *


@pytest.fixture(params=["x", "y", "both"])
def direction(request):
    return request.param


@pytest.fixture(params=[False, True],
                ids=["tri", "quad"])
def quadrilateral(request):
    return request.param


@pytest.fixture(params=["left", "right", "crossed"])
def diagonal(request):
    return request.param


def run_periodic_helmholtz(direction, quadrilateral, diagonal):
    if quadrilateral:
        if diagonal == "left":
            # run the test
            diagonal = None
        else:
            # don't run the test
            return

    mesh = PeriodicRectangleMesh(100, 60, 5, 3, quadrilateral=quadrilateral,
                                 diagonal=diagonal, direction=direction)
    x = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)

    u_exact = Function(V)
    u_exact.interpolate(sin(4.0*pi*x[0]/5.0)*sin(2.0*pi*x[1]/3.0))

    f = Function(V).assign((244.0*pi*pi/225.0 + 1.0)*u_exact)

    if direction in ("x", "y"):
        bcs = DirichletBC(V, Constant(0), (1, 2))
    elif direction == "both":
        bcs = []
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    L = inner(f, v)*dx

    out = Function(V)
    solve(a == L, out, solver_parameters={'ksp_type': 'cg'}, bcs=bcs)

    l2err = sqrt(assemble(inner((out-u_exact), (out-u_exact))*dx))
    l2norm = sqrt(assemble(inner(u_exact, u_exact)*dx))
    assert l2err/l2norm < 0.004


def test_periodic_helmholtz(direction, quadrilateral, diagonal):
    run_periodic_helmholtz(direction, quadrilateral, diagonal)


@pytest.mark.parallel(nprocs=3)
def test_periodic_helmholtz_parallel(direction, quadrilateral, diagonal):
    run_periodic_helmholtz(direction, quadrilateral, diagonal)
