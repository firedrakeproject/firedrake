"""This demo program sets the top, bottom and side boundaries
of an extruded unit square. We then check against the actual solution
of the equation.
"""
import pytest
from firedrake import *


def run_test_3D(size, parameters={}, test_mode=False):
    # Create mesh and define function space
    m = UnitSquareMesh(size, size)
    layers = size + 1
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / (layers - 1))

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    exp = Expression('x[0]*x[0] - x[1]*x[1] - x[2]*x[2]')
    bcs = [DirichletBC(V, exp, "bottom"),
           DirichletBC(V, exp, "top"),
           DirichletBC(V, exp, 1),
           DirichletBC(V, exp, 2),
           DirichletBC(V, exp, 3),
           DirichletBC(V, exp, 4)]

    v = TestFunction(V)
    u = TrialFunction(V)
    a = dot(grad(u), grad(v)) * dx

    f = Function(V)
    f.assign(2)

    L = v * f * dx

    out = Function(V)

    exact = Function(V)
    exact.interpolate(exp)

    solve(a == L, out, bcs=bcs)

    res = sqrt(assemble(dot(out - exact, out - exact) * dx))

    if not test_mode:
        print "The error is ", res
        file = File("side-bcs-computed.pvd")
        file << out
        file = File("side-bcs-expected.pvd")
        file << exact
    return res


def run_test_2D(intervals, parameters={}, test_mode=False):
    # Create mesh and define function space
    m = UnitIntervalMesh(intervals)
    layers = intervals+1
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / (layers - 1))

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    exp = Expression('x[0]*x[0] - 2*x[1]*x[1]')
    bcs = [DirichletBC(V, exp, "bottom"),
           DirichletBC(V, exp, "top"),
           DirichletBC(V, exp, 1),
           DirichletBC(V, exp, 2)]

    v = TestFunction(V)
    u = TrialFunction(V)
    a = dot(grad(u), grad(v)) * dx

    f = Function(V)
    f.assign(2)

    L = v * f * dx

    out = Function(V)

    exact = Function(V)
    exact.interpolate(exp)

    solve(a == L, out, bcs=bcs)

    res = sqrt(assemble(dot(out - exact, out - exact) * dx))

    if not test_mode:
        print "The error is ", res
        file = File("side-bcs-computed.pvd")
        file << out
        file = File("side-bcs-expected.pvd")
        file << exact
    return res


def test_extrusion_side_strong_bcs():
    assert (run_test_3D(3, test_mode=True) < 1.e-13)


def test_extrusion_side_strong_bcs_large():
    assert (run_test_3D(6, test_mode=True) < 1.e-08)


def test_extrusion_side_strong_bcs_2D():
    assert (run_test_2D(2, test_mode=True) < 1.e-13)


def test_extrusion_side_strong_bcs_2D_large():
    assert (run_test_2D(4, test_mode=True) < 1.e-12)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
