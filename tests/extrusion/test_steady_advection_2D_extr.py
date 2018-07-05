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


@pytest.fixture(scope='module')
def DG0(mesh):
    return FunctionSpace(mesh, "DG", 0)


@pytest.fixture(scope='module')
def DG1(mesh):
    return FunctionSpace(mesh, "DG", 1)


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
    a2 = jump(phi)*(un('+')*D('+') - un('-')*D('-'))*dS_v
    a3 = phi*un*D*ds_v(2)  # outflow at right-hand wall
    a = a1 + a2 + a3

    L = -inflow*phi*dot(u0, n)*ds_v(1)  # inflow at left-hand wall

    out = Function(DG0)
    solve(a == L, out)

    # we only use inflow at the left wall, but since the velocity field
    # is parallel to the coordinate axis, the exact solution matches
    # the inflow function
    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-14


def test_left_to_right(mesh, DG0, W):
    run_left_to_right(mesh, DG0, W)


@pytest.mark.parallel
def test_left_to_right_parallel():
    m = mesh()
    dg0 = DG0(m)
    w = W(m)
    run_left_to_right(m, dg0, w)


def run_right_to_left(mesh, DG1, W):
    velocity = Expression(("-1.0", "0.0"))
    u0 = project(velocity, W)

    inflowexpr = Expression("x[1] + 0.5")
    inflow = Function(DG1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DG1)
    phi = TestFunction(DG1)

    a1 = -D*dot(u0, grad(phi))*dx
    a2 = jump(phi)*(un('+')*D('+') - un('-')*D('-'))*dS_v
    a3 = phi*un*D*ds_v(1)  # outflow at left-hand wall
    a = a1 + a2 + a3

    L = -inflow*phi*dot(u0, n)*ds_v(2)  # inflow at right-hand wall

    out = Function(DG1)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 2e-14


def test_right_to_left(mesh, DG1, W):
    run_right_to_left(mesh, DG1, W)


@pytest.mark.parallel
def test_right_to_left_parallel():
    m = mesh()
    dg1 = DG1(m)
    w = W(m)
    run_right_to_left(m, dg1, w)


def run_bottom_to_top(mesh, DG0, W):
    velocity = Expression(("0.0", "1.0"))
    u0 = project(velocity, W)

    inflowexpr = Expression("if(x[0] > 0.25 and x[0] < 0.75, 1, 0.5)")
    inflow = Function(DG0)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DG0)
    phi = TestFunction(DG0)

    a1 = -D*dot(u0, grad(phi))*dx
    a2 = jump(phi)*(un('+')*D('+') - un('-')*D('-'))*dS_h
    a3 = phi*un*D*ds_t  # outflow at top wall
    a = a1 + a2 + a3

    L = -inflow*phi*dot(u0, n)*ds_b  # inflow at bottom wall

    out = Function(DG0)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-14


def test_bottom_to_top(mesh, DG0, W):
    run_bottom_to_top(mesh, DG0, W)


@pytest.mark.parallel
def test_bottom_to_top_parallel():
    m = mesh()
    dg0 = DG0(m)
    w = W(m)
    run_bottom_to_top(m, dg0, w)


def run_top_to_bottom(mesh, DG1, W):
    velocity = Expression(("0.0", "-1.0"))
    u0 = project(velocity, W)

    inflowexpr = Expression("0.5 + fabs(x[0] - 0.5)")
    inflow = Function(DG1)
    inflow.interpolate(inflowexpr)

    n = FacetNormal(mesh)
    un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

    D = TrialFunction(DG1)
    phi = TestFunction(DG1)

    a1 = -D*dot(u0, grad(phi))*dx
    a2 = jump(phi)*(un('+')*D('+') - un('-')*D('-'))*dS_h
    a3 = phi*un*D*ds_b  # outflow at bottom wall
    a = a1 + a2 + a3

    L = -inflow*phi*dot(u0, n)*ds_t  # inflow at top wall

    out = Function(DG1)
    solve(a == L, out)

    assert max(abs(out.dat.data - inflow.dat.data)) < 1e-14


def test_top_to_bottom(mesh, DG1, W):
    run_top_to_bottom(mesh, DG1, W)


@pytest.mark.parallel
def test_top_to_bottom_parallel():
    m = mesh()
    dg1 = DG1(m)
    w = W(m)
    run_top_to_bottom(m, dg1, w)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
