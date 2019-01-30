# Simple Poisson equation
# =========================

import pytest

from firedrake import *


def test_nonlinear_EquationBC():

    mesh = UnitSquareMesh(20, 20)

    V = FunctionSpace(mesh, "CG", 3)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(- 8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2))

    a = - dot(grad(v), grad(u)) * dx
    L = f * v * dx

    g1 = Function(V)
    g1.interpolate(cos(2 * pi * y))

    g3 = Function(V)
    g3.interpolate(cos(2 * pi * x))

    bc1 = EquationBC(v * (u - g1) * ds(1) == 0, u, 1)
    bc2 = DirichletBC(V, cos(2 * pi * y), 2)
    bc3 = EquationBC(v * (u - g3) * ds(3) == 0, u, 3)
    bc4 = DirichletBC(V, cos(2 * pi * x), 4)

    solve(a - L == 0, u, bcs=[bc1, bc2, bc3, bc4], solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1e-12, 'ksp_rtol': 1e-20, 'ksp_divtol': 1e8})

    f.interpolate(cos(x * pi * 2)*cos(y * pi * 2))
    err = sqrt(assemble(dot(u - f, u - f) * dx))

    assert(err < 3.e-5)


def test_linear_EquationBC():

    mesh = UnitSquareMesh(20, 20)

    V = FunctionSpace(mesh, "CG", 3)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(- 8.0 * pi * pi * cos(x * pi * 2)*cos(y * pi * 2))

    a = - dot(grad(v), grad(u)) * dx
    L = f * v * dx

    g1 = Function(V)
    g1.interpolate(cos(2 * pi * y))

    g3 = Function(V)
    g3.interpolate(cos(2 * pi * x))

    u_ = Function(V)

    bc1 = EquationBC(v * u * ds(1) == v * g1 * ds(1), u_, 1)
    bc2 = DirichletBC(V, cos(2 * pi * y), 2)
    bc3 = EquationBC(v * u * ds(3) == v * g3 * ds(3), u_, 3)
    bc4 = DirichletBC(V, cos(2 * pi * x), 4)

    solve(a == L, u_, bcs=[bc1, bc2, bc3, bc4], solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1e-12, 'ksp_rtol': 1e-20, 'ksp_divtol': 1e8})

    f.interpolate(cos(x * pi * 2) * cos(y * pi * 2))
    err = sqrt(assemble(dot(u_ - f, u_ - f) * dx))

    assert(err < 3.e-5)
