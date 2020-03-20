import pytest
from firedrake import *


def test_filter_one_form():

    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)

    tV = FunctionSpace(mesh.topology, 'CG', 1)
    fltr = CoordinatelessFunction(tV)

    x, y = SpatialCoordinate(mesh)

    f = Function(V).interpolate(8.0 * pi * pi * cos(2 * pi *x) * cos(2 * pi * y))

    rhs = assemble(f * v * dx(degree=4))
    #rhs = assemble(f * Filtered(v, fltr) * dx(degree=4))
    #bc = DirichletBC(V, g, 1)


    print(rhs.dat.data)

"""
def test_filter_poisson():

    mesh = UnitSquareMesh(2, 2)

    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    u = TrialFunction(V)

    tV = FunctionSpace(mesh.topology, 'CG', 1)
    fltr = CoordinatelessFunction(tV)

    x, y = SpatialCoordinate(mesh)

    # Analytical solution
    g = Function(V).interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    f = Function(V).interpolate(8.0 * pi * pi * cos(2 * pi *x) * cos(2 * pi * y))

    a = dot(grad(v), grad(u)) * dx
    L = f * v * dx

    bc = DirichletBC(V, g, 1)

    u = Function(V)

    solve(a == L, u, bcs = [bc, ], solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    assert(sqrt(assemble(dot(u - g, u - g) * dx)) < 1e-19)
"""


