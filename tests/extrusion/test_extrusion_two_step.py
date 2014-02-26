"""Testing extruded RT elements."""

import pytest

from firedrake import *


def two_step():
    power = 4
    # Create mesh and define function space
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 10

    # Populate the coordinates of the extruded mesh by providing the
    # coordinates as a field.

    mesh = ExtrudedMesh(m, layers, layer_height=0.1)

    V = FunctionSpace(mesh, "Lagrange", 2, vfamily="DG", vdegree=0)

    horiz = FiniteElement("BDM", "triangle", 1)
    vert = FiniteElement("DG", "interval", 0)
    prod = HDiv(OuterProductElement(horiz, vert))
    W = FunctionSpace(mesh, prod)
    X = FunctionSpace(mesh, "DG", 0, vfamily="DG", vdegree=0)

    # Define starting field
    f0 = Function(V)
    f0.interpolate(Expression("1 + x[0]*x[0] + x[1]*x[1]"))

    # DO IN ONE STEP
    u = TrialFunction(X)
    v = TestFunction(X)
    a = u * v * dx
    L = div(grad(f0)) * v * dx

    assemble(a)
    assemble(L)
    f_e = Function(X)
    solve(a == L, f_e)

    # DO IN TWO STEPS
    u = TrialFunction(W)
    v = TestFunction(W)
    a = dot(u, v) * dx
    L = dot(grad(f0), v) * dx

    # Compute solution
    assemble(a)
    assemble(L)
    f1 = Function(W)
    solve(a == L, f1)
    # FIXME x should be (2x, 2y) but we have no way of checking

    u = TrialFunction(X)
    v = TestFunction(X)
    a = u * v * dx
    L = div(f1) * v * dx

    # Compute solution
    assemble(a)
    assemble(L)
    f2 = Function(X)
    solve(a == L, f2)

    return np.max(np.abs(f2.dat.data - f_e.dat.data))


def test_firedrake_extrusion_two_step():
    assert two_step() < 1.0e-4

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
