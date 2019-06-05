"""Testing extruded RT elements."""

import numpy as np
from firedrake import *


def two_step(quadrilateral):
    power = 4
    # Create mesh and define function space
    m = UnitSquareMesh(2 ** power, 2 ** power, quadrilateral=quadrilateral)
    layers = 10

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.

    mesh = ExtrudedMesh(m, layers, layer_height=0.1)

    V = FunctionSpace(mesh, "Lagrange", 2, vfamily="DG", vdegree=0)

    if quadrilateral:
        horiz = FiniteElement("RTCF", "quadrilateral", 1)
    else:
        horiz = FiniteElement("BDM", "triangle", 1)
    vert = FiniteElement("DG", "interval", 0)
    prod = HDiv(TensorProductElement(horiz, vert))
    W = FunctionSpace(mesh, prod)
    X = FunctionSpace(mesh, "DG", 0, vfamily="DG", vdegree=0)

    # Define starting field
    xs = SpatialCoordinate(mesh)
    f0 = Function(V)
    f0.interpolate(1 + xs[0]*xs[0] + xs[1]*xs[1])

    # DO IN ONE STEP
    u = TrialFunction(X)
    v = TestFunction(X)
    a = inner(u, v) * dx
    L = inner(div(grad(f0)), v) * dx

    assemble(a)
    assemble(L)
    f_e = Function(X)
    solve(a == L, f_e)

    # DO IN TWO STEPS
    u = TrialFunction(W)
    v = TestFunction(W)
    a = inner(u, v) * dx
    L = inner(grad(f0), v) * dx

    # Compute solution
    assemble(a)
    assemble(L)
    f1 = Function(W)
    solve(a == L, f1)
    # FIXME x should be (2x, 2y) but we have no way of checking

    u = TrialFunction(X)
    v = TestFunction(X)
    a = inner(u, v) * dx
    L = inner(div(f1), v) * dx

    # Compute solution
    assemble(a)
    assemble(L)
    f2 = Function(X)
    solve(a == L, f2)

    return np.max(np.abs(f2.dat.data - f_e.dat.data))


def test_firedrake_extrusion_two_step():
    assert two_step(quadrilateral=False) < 1.0e-4


def test_firedrake_extrusion_two_step_quadrilateral():
    assert two_step(quadrilateral=True) < 1.0e-4
