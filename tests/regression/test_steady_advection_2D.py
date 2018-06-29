"""This demo program solves the steady-state advection equation
div(u0*D) = 0, for a prescribed velocity field u0.  An upwind
method is used, which stress-tests both interior and exterior
facet integrals.
"""

import pytest
from firedrake import *


@pytest.fixture(params=[False, True],
                ids=["triangle", "quadrilateral"])
def mesh(request):
    return UnitSquareMesh(5, 5, quadrilateral=request.param)


@pytest.fixture
def DG0(mesh):
    return FunctionSpace(mesh, "DG", 0)


@pytest.fixture
def DG1(mesh):
    return FunctionSpace(mesh, "DG", 1)


@pytest.fixture
def W(mesh):
    if mesh.ufl_cell() == triangle:
        return FunctionSpace(mesh, "BDM", 1)
    else:
        return FunctionSpace(mesh, "RTCF", 1)


def run_left_to_right(mesh, DG0, W):
    velocity = Expression(("1.0", "0.0"))
    u0 = project(velocity, W)

    inflowexpr = Expression("if(x[1] > 0.25 and x[1] < 0.75, 1, 0.5)")
    inflow = Function(DG0)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DG0)
    phi = TestFunction(DG0)

    a1 = -D*dot(u0, grad(phi))*dx
    a2 = jump(phi)*(un('+')*D('+') - un('-')*D('-'))*dS
    a3 = phi*un*D*ds(2)  # outflow at right-hand wall
    a = a1 + a2 + a3

    L = -inflow*phi*dot(u0, n)*ds(1)  # inflow at left-hand wall

    out = Function(DG0)
    solve(a == L, out)

    # we only use inflow at the left wall, but since the velocity field
    # is parallel to the coordinate axis, the exact solution matches
    # the inflow function
    assert max(abs(out.dat.data - inflow.dat.data)) < 1.2e-7


def test_left_to_right(mesh, DG0, W):
    run_left_to_right(mesh, DG0, W)


@pytest.mark.parallel
def test_left_to_right_parallel(mesh, DG0, W):
    run_left_to_right(mesh, DG0, W)


def run_up_to_down(mesh, DG1, W):
    velocity = Expression(("0.0", "-1.0"))
    u0 = project(velocity, W)

    inflowexpr = Expression("1.0 + x[0]")
    inflow = Function(DG1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DG1)
    phi = TestFunction(DG1)

    a1 = -D*dot(u0, grad(phi))*dx
    a2 = jump(phi)*(un('+')*D('+') - un('-')*D('-'))*dS
    a3 = phi*un*D*ds(3)  # outflow at lower wall
    a = a1 + a2 + a3

    L = -inflow*phi*dot(u0, n)*ds(4)  # inflow at upper wall

    out = Function(DG1)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1.1e-6


def test_up_to_down(mesh, DG1, W):
    run_up_to_down(mesh, DG1, W)


@pytest.mark.parallel
def test_up_to_down_parallel(mesh, DG1, W):
    run_up_to_down(mesh, DG1, W)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
