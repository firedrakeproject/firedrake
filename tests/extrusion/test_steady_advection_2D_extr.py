"""This demo program solves the steady-state advection equation
div(u0*D) = 0, for a prescribed velocity field u0.  An upwind
method is used, which stress-tests both interior and exterior
facet integrals.
"""

import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    m = UnitIntervalMesh(10)
    return ExtrudedMesh(m, layers=4, layer_height=0.25)


@pytest.fixture(scope='module', params=["DG", "DPC"])
def DGDPC0(request, mesh):
    return FunctionSpace(mesh, request.param, 0)


@pytest.fixture(scope='module', params=["DG", "DPC"])
def DGDPC1(request, mesh):
    return FunctionSpace(mesh, request.param, 1)


@pytest.fixture
def W(mesh):
    # BDM1 element on a quad
    W0_h = FiniteElement("CG", "interval", 1)
    W0_v = FiniteElement("DG", "interval", 1)
    W0 = HDiv(TensorProductElement(W0_h, W0_v))

    W1_h = FiniteElement("DG", "interval", 1)
    W1_v = FiniteElement("CG", "interval", 1)
    W1 = HDiv(TensorProductElement(W1_h, W1_v))

    return FunctionSpace(mesh, W0+W1)


def run_left_to_right(mesh, DGDPC0, W):
    velocity = as_vector([1.0, 0.0])
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)

    inflowexpr = conditional(And(real(xs[1]) > 0.25, real(xs[1]) < 0.75), 1.0, 0.5)
    inflow = Function(DGDPC0)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC0)
    phi = TestFunction(DGDPC0)

    a1 = inner(-D, dot(u0, grad(phi)))*dx
    a2 = inner((un('+')*D('+') - un('-')*D('-')), jump(phi))*dS_v
    a3 = un*inner(D, phi)*ds_v(2)  # outflow at right-hand wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_v(1)  # inflow at left-hand wall

    out = Function(DGDPC0)
    solve(a == L, out)

    # we only use inflow at the left wall, but since the velocity field
    # is parallel to the coordinate axis, the exact solution matches
    # the inflow function
    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-14


def test_left_to_right(mesh, DGDPC0, W):
    run_left_to_right(mesh, DGDPC0, W)


@pytest.mark.parallel
def test_left_to_right_parallel(mesh, DGDPC0, W):
    run_left_to_right(mesh, DGDPC0, W)


def run_right_to_left(mesh, DGDPC1, W):
    velocity = as_vector([-1.0, 0.0])
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = xs[1] + 0.5
    inflow = Function(DGDPC1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC1)
    phi = TestFunction(DGDPC1)

    a1 = -inner(D, dot(u0, grad(phi)))*dx
    a2 = inner((un('+')*D('+') - un('-')*D('-')), jump(phi))*dS_v
    a3 = un*inner(D, phi)*ds_v(1)  # outflow at left-hand wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_v(2)  # inflow at right-hand wall

    out = Function(DGDPC1)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 2e-14


def test_right_to_left(mesh, DGDPC1, W):
    run_right_to_left(mesh, DGDPC1, W)


@pytest.mark.parallel
def test_right_to_left_parallel(mesh, DGDPC1, W):
    run_right_to_left(mesh, DGDPC1, W)


def run_bottom_to_top(mesh, DGDPC0, W):
    velocity = as_vector([0.0, 1.0])
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = conditional(And(real(xs[0]) > 0.25, real(xs[0]) < 0.75), 1.0, 0.5)
    inflow = Function(DGDPC0)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC0)
    phi = TestFunction(DGDPC0)

    a1 = -inner(D, dot(u0, grad(phi)))*dx
    a2 = inner((un('+')*D('+') - un('-')*D('-')), jump(phi))*dS_h
    a3 = inner(D*un, phi)*ds_t  # outflow at top wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_b  # inflow at bottom wall

    out = Function(DGDPC0)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-14


def test_bottom_to_top(mesh, DGDPC0, W):
    run_bottom_to_top(mesh, DGDPC0, W)


@pytest.mark.parallel
def test_bottom_to_top_parallel(mesh, DGDPC0, W):
    run_bottom_to_top(mesh, DGDPC0, W)


def run_top_to_bottom(mesh, DGDPC1, W):
    velocity = as_vector([0.0, -1.0])
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = 0.5 + abs(xs[0] - 0.5)
    inflow = Function(DGDPC1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC1)
    phi = TestFunction(DGDPC1)

    a1 = -inner(D, dot(u0, grad(phi)))*dx
    a2 = inner((un('+')*D('+') - un('-')*D('-')), jump(phi))*dS_h
    a3 = inner(D, phi*un)*ds_b  # outflow at bottom wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_t  # inflow at top wall

    out = Function(DGDPC1)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-14


def test_top_to_bottom(mesh, DGDPC1, W):
    run_top_to_bottom(mesh, DGDPC1, W)


@pytest.mark.parallel
def test_top_to_bottom_parallel(mesh, DGDPC1, W):
    run_top_to_bottom(mesh, DGDPC1, W)
