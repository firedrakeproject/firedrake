"""This demo solves Poisson's equation

  - div grad u(x, y) = 0

on an extruded unit cube with boundary conditions given by:

  u(x, y, 0) = 0
  v(x, y, 1) = 42

Homogeneous Neumann boundary conditions are applied naturally on the
other sides of the domain.

This has the analytical solution

  u(x, y, z) = 42*z

"""
import pytest
from firedrake import *


def run_test(layers):
    # Create mesh and define function space
    m = UnitSquareMesh(1, 1)
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / (layers - 1))

    V = FunctionSpace(mesh, "CG", 1)
    bcs = [DirichletBC(V, 0, "bottom"),
           DirichletBC(V, 42, "top")]

    v = TestFunction(V)
    u = TrialFunction(V)
    a = dot(grad(u), grad(v)) * dx
    f = Function(V)
    f.assign(0)
    L = v * f * dx
    u = Function(V)
    exact = Function(V)
    exact.interpolate(Expression('42*x[2]'))
    solve(a == L, u, bcs=bcs)
    res = sqrt(assemble(dot(u - exact, u - exact) * dx))
    return res


def test_extrusion_poisson_strong_bcs():
    for layers in [2, 3, 11]:
        assert (run_test(layers) < 1.e-6)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
