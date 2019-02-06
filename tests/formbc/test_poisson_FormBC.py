# Simple Poisson equation
# =========================

import pytest

from firedrake import *
import math


def nonlinear_poisson(mat_type, mesh_num, porder):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    V = FunctionSpace(mesh, "CG", porder)

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

    solve(a - L == 0, u, bcs=[bc1], solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1e-10, 'ksp_rtol': 1e-10, 'ksp_max_it': 200000, 'ksp_divtol': 1e8, 'mat_type': mat_type})

    f.interpolate(cos(x * pi * 2)*cos(y * pi * 2))
    return sqrt(assemble(dot(u - f, u - f) * dx))


def linear_poisson(mat_type, mesh_num, porder):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    V = FunctionSpace(mesh, "CG", porder)

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

    solve(a == L, u_, bcs=[bc1], solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1e-10, 'ksp_rtol': 1e-10, 'ksp_divtol': 1e8, 'ksp_max_it': 100000, 'mat_type': mat_type})

    f.interpolate(cos(x * pi * 2) * cos(y * pi * 2))
    return sqrt(assemble(dot(u_ - f, u_ - f) * dx))


@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
@pytest.mark.parametrize("porder", [3])
def test_nonlinear_EquationBC(mat_type, porder):

    err=[]
    for mesh_num in [8, 16]:
        err.append(nonlinear_poisson(mat_type, mesh_num, porder))

    assert(abs(math.log2(err[0]) - math.log2(err[1]) - (porder+1)) < 0.01)


@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
@pytest.mark.parametrize("porder", [3])
def test_linear_EquationBC(mat_type, porder):

    err=[]
    for mesh_num in [8, 16]:
        err.append(linear_poisson(mat_type, mesh_num, porder))

    assert(abs(math.log2(err[0]) - math.log2(err[1]) - (porder+1)) < 0.01)
