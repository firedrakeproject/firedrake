"""This demo program solves the steady-state advection equation
div(u0*D) = 0, for a prescribed velocity field u0.  An upwind
method is used, which stress-tests both interior and exterior
facet integrals.
"""

import pytest
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    quadrilateral = request.param
    m = UnitSquareMesh(4, 4, quadrilateral=quadrilateral)
    return ExtrudedMesh(m, layers=4, layer_height=0.25)


@pytest.fixture(scope='module', params=["DG", "DPC"])
def DGDPC0(request, mesh):
    if mesh._base_mesh.ufl_cell() == triangle:
        return FunctionSpace(mesh, "DG", 0)
    else:
        return FunctionSpace(mesh, request.param, 0)


@pytest.fixture(scope='module', params=["DG", "DPC"])
def DGDPC1(request, mesh):
    if mesh._base_mesh.ufl_cell() == triangle:
        return FunctionSpace(mesh, "DG", 1)
    else:
        return FunctionSpace(mesh, request.param, 1)


@pytest.fixture
def W(mesh):
    if mesh.ufl_cell().sub_cells()[0].cellname() == "quadrilateral":
        # RTCF1 element on a hexahedron
        W0_h = FiniteElement("RTCF", "quadrilateral", 1)
        W1_h = FiniteElement("DQ", "quadrilateral", 0)
    else:
        # BDM1 element on a prism
        W0_h = FiniteElement("BDM", "triangle", 1)
        W1_h = FiniteElement("DG", "triangle", 0)

    W0_v = FiniteElement("DG", "interval", 0)
    W0 = HDiv(TensorProductElement(W0_h, W0_v))

    W1_v = FiniteElement("CG", "interval", 1)
    W1 = HDiv(TensorProductElement(W1_h, W1_v))

    return FunctionSpace(mesh, W0+W1)


def test_left_to_right(mesh, DGDPC1, W):
    velocity = as_vector((1.0, 0.0, 0.0))
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = xs[1] + xs[2]
    inflow = Function(DGDPC1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC1)
    phi = TestFunction(DGDPC1)

    a1 = -inner(D, dot(u0, grad(phi)))*dx
    a2 = inner(un('+')*D('+') - un('-')*D('-'), jump(phi))*dS_v
    a3 = inner(D*un, phi)*ds_v(2)  # outflow at right-hand wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_v(1)  # inflow at left-hand wall

    out = Function(DGDPC1)
    solve(a == L, out)

    # we only use inflow at the left wall, but since the velocity field
    # is parallel to the coordinate axis, the exact solution matches
    # the inflow function
    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-6


def test_right_to_left(mesh, DGDPC0, W):
    velocity = as_vector((-1.0, 0.0, 0.0))
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = conditional(And(real(xs[1]) > 0.25, real(xs[1]) < 0.75), 1.0, 0.5)
    inflow = Function(DGDPC0)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC0)
    phi = TestFunction(DGDPC0)

    a1 = -inner(D, dot(u0, grad(phi)))*dx
    a2 = inner((un('+')*D('+') - un('-')*D('-')), jump(phi))*dS_v
    a3 = inner(un*D, phi)*ds_v(1)  # outflow at left-hand wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_v(2)  # inflow at right-hand wall

    out = Function(DGDPC0)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-7


def test_near_to_far(mesh, DGDPC1, W):
    velocity = as_vector((0.0, 1.0, 0.0))
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = 0.5 + abs(xs[2] - 0.5)
    inflow = Function(DGDPC1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC1)
    phi = TestFunction(DGDPC1)

    a1 = -inner(D, dot(u0, grad(phi)))*dx
    a2 = inner((un('+')*D('+') - un('-')*D('-')), jump(phi))*dS_v
    a3 = inner(un*D, phi)*ds_v(4)  # outflow at far wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_v(3)  # inflow at near wall

    out = Function(DGDPC1)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 3.5e-7


def test_far_to_near(mesh, DGDPC0, W):
    velocity = as_vector((0.0, -1.0, 0.0))
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = conditional(And(real(xs[2]) > 0.25, real(xs[2]) < 0.75), 1.0, 0.5)
    inflow = Function(DGDPC0)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC0)
    phi = TestFunction(DGDPC0)

    a1 = -inner(D, dot(u0, grad(phi)))*dx
    a2 = inner(un('+')*D('+') - un('-')*D('-'), jump(phi))*dS_v
    a3 = inner(un*D, phi)*ds_v(3)  # outflow at near wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_v(4)  # inflow at far wall

    out = Function(DGDPC0)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1.4e-7


def test_bottom_to_top(mesh, DGDPC1, W):
    velocity = as_vector((0.0, 0.0, 1.0))
    u0 = project(velocity, W)

    xs = SpatialCoordinate(mesh)
    inflowexpr = 0.5 + xs[0]
    inflow = Function(DGDPC1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DGDPC1)
    phi = TestFunction(DGDPC1)

    a1 = -inner(D, dot(u0, grad(phi)))*dx
    a2 = inner(un('+')*D('+') - un('-')*D('-'), jump(phi))*dS_h
    a3 = inner(un*D, phi)*ds_t  # outflow at top wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_b  # inflow at bottom wall

    out = Function(DGDPC1)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-13


def test_top_to_bottom(mesh, DGDPC0, W):
    velocity = as_vector((0.0, 0.0, -1.0))
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
    a2 = inner(un('+')*D('+') - un('-')*D('-'), jump(phi))*dS_h
    a3 = inner(un*D, phi)*ds_b  # outflow at bottom wall
    a = a1 + a2 + a3

    L = -inner(inflow*dot(u0, n), phi)*ds_t  # inflow at top wall

    out = Function(DGDPC0)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-14
