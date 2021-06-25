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
from firedrake import *


def run_test(layers, quadrilateral):
    # Create mesh and define function space
    m = UnitSquareMesh(1, 1, quadrilateral=quadrilateral)
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / layers)

    V = FunctionSpace(mesh, "CG", 1)
    bcs = [DirichletBC(V, 0, "bottom"),
           DirichletBC(V, 42, "top")]

    v = TestFunction(V)
    u = TrialFunction(V)
    a = inner(grad(u), grad(v)) * dx
    f = Function(V)
    f.assign(0)
    L = inner(f, v) * dx
    u = Function(V)
    exact = Function(V)
    xs = SpatialCoordinate(mesh)
    exact.interpolate(42*xs[2])
    solve(a == L, u, bcs=bcs)
    res = sqrt(assemble(inner(u - exact, u - exact) * dx))
    return res


def test_extrusion_poisson_strong_bcs():
    for layers in [1, 2, 10]:
        assert (run_test(layers, quadrilateral=False) < 1.e-6)


def test_extrusion_poisson_strong_bcs_quadrilateral():
    for layers in [1, 2, 10]:
        assert (run_test(layers, quadrilateral=True) < 1.e-6)
