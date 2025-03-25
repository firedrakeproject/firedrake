"""This demo program solves the steady-state advection equation
div(u0*D) = 0, for a prescribed velocity field u0.  An upwind
method is used, which stress-tests both interior and exterior
facet integrals.
"""

import pytest
from firedrake import *


@pytest.fixture(scope='module', params=[False, True],
                ids=["triangle", "quadrilateral"])
def mesh(request):
    return UnitSquareMesh(5, 5, quadrilateral=request.param)


@pytest.fixture(scope='module', params=["DG", "DPC"])
def DGDPC0(request, mesh):
    if mesh.ufl_cell() == triangle:
        return FunctionSpace(mesh, "DG", 0)
    else:
        return FunctionSpace(mesh, request.param, 0)


@pytest.fixture(scope='module', params=["DG", "DPC"])
def DGDPC1(request, mesh):
    if mesh.ufl_cell() == triangle:
        return FunctionSpace(mesh, "DG", 1)
    else:
        return FunctionSpace(mesh, request.param, 1)


@pytest.fixture(scope='module')
def W(mesh):
    if mesh.ufl_cell() == triangle:
        return FunctionSpace(mesh, "BDM", 1)
    else:
        return FunctionSpace(mesh, "RTCF", 1)


def run_left_to_right(mesh, DGDPC0, W):
    velocity = as_vector((1.0, 0.0))
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = conditional(And(real(xs[1]) > 0.25, real(xs[1]) < 0.75), 1.0, 0.5)
    inflow = Function(DGDPC0)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC0)
    phi = TestFunction(DGDPC0)

    a1 = -D * inner(u0, grad(phi)) * dx
    a2 = inner(un('+')*D('+') - un('-')*D('-'), jump(phi)) * dS
    a3 = inner(un*D, phi) * ds(2)  # outflow at right-hand wall
    a = a1 + a2 + a3

    L = -inflow * inner(dot(u0, n), phi) * ds(1)  # inflow at left-hand wall

    out = Function(DGDPC0)
    solve(a == L, out)

    # we only use inflow at the left wall, but since the velocity field
    # is parallel to the coordinate axis, the exact solution matches
    # the inflow function
    assert max(abs(out.dat.data - inflow.dat.data)) < 1.2e-7


def test_left_to_right(mesh, DGDPC0, W):
    run_left_to_right(mesh, DGDPC0, W)


@pytest.mark.parallel
def test_left_to_right_parallel(mesh, DGDPC0, W):
    run_left_to_right(mesh, DGDPC0, W)


def run_up_to_down(mesh, DGDPC1, W):
    velocity = as_vector((0.0, -1.0))
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = 1 + xs[0]
    inflow = Function(DGDPC1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC1)
    phi = TestFunction(DGDPC1)

    a1 = -D * inner(u0, grad(phi)) * dx
    a2 = inner(un('+')*D('+') - un('-')*D('-'), jump(phi)) * dS
    a3 = inner(un*D, phi) * ds(3)  # outflow at lower wall
    a = a1 + a2 + a3

    L = -inflow * inner(dot(u0, n), phi) * ds(4)  # inflow at upper wall

    out = Function(DGDPC1)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1.1e-6


def test_up_to_down(mesh, DGDPC1, W):
    run_up_to_down(mesh, DGDPC1, W)


@pytest.mark.parallel
def test_up_to_down_parallel(mesh, DGDPC1, W):
    run_up_to_down(mesh, DGDPC1, W)
