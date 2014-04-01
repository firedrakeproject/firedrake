"""This demo program solves the steady-state advection equation
div(u0*D) = 0, for a prescribed velocity field u0.  An upwind
method is used, which stress-tests both interior and exterior
facet integrals.
"""

import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitCubeMesh(3, 3, 3)


@pytest.fixture(scope='module')
def DG0(mesh):
    return FunctionSpace(mesh, "DG", 0)


@pytest.fixture(scope='module')
def DG1(mesh):
    return FunctionSpace(mesh, "DG", 1)


@pytest.fixture(scope='module')
def W(mesh):
    return FunctionSpace(mesh, "RT", 1)


def test_3d_near_to_far(mesh, DG0, W):
    velocity = Expression(("0.0", "1.0", "0.0"))
    u0 = project(velocity, W)

    inflowexpr = Expression("x[2] > 0.33 && x[2] < 0.67 ? 1.0 : 0.5")
    inflow = Function(DG0)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DG0)
    phi = TestFunction(DG0)

    a1 = -D*dot(u0, grad(phi))*dx
    a2 = dot(jump(phi), un('+')*D('+') - un('-')*D('-'))*dS
    a3 = dot(phi, un*D)*ds(4)  # outflow at far wall
    a = a1 + a2 + a3

    L = -inflow*phi*dot(u0, n)*ds(3)  # inflow at near wall

    out = Function(DG0)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-6


def test_3d_up_to_down(mesh, DG1, W):
    velocity = Expression(("0.0", "0.0", "-1.0"))
    u0 = project(velocity, W)

    inflowexpr = Expression("x[0] + x[1]")
    inflow = Function(DG1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DG1)
    phi = TestFunction(DG1)

    a1 = -D*dot(u0, grad(phi))*dx
    a2 = dot(jump(phi), un('+')*D('+') - un('-')*D('-'))*dS
    a3 = dot(phi, un*D)*ds(5)  # outflow at lower wall
    a = a1 + a2 + a3

    L = -inflow*phi*dot(u0, n)*ds(6)  # inflow at upper wall

    out = Function(DG1)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1.3e-6

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
