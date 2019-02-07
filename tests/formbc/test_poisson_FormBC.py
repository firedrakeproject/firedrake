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

    solve(a == L, u_, bcs=[bc1], solver_parameters={'ksp_type': 'gmres', 'ksp_atol': 1e-10, 'ksp_rtol': 1e-10, 'ksp_divtol': 1e8, 'ksp_max_it': 100000, 'mat_type': mat_type})

    f.interpolate(cos(x * pi * 2) * cos(y * pi * 2))
    return sqrt(assemble(dot(u_ - f, u_ - f) * dx))


def nonlinear_poisson_mixed(mat_type, mesh_num, porder):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    BDM = FunctionSpace(mesh, "BDM", porder+1)
    DG = FunctionSpace(mesh, "DG", porder)
    W = BDM * DG

    w = Function(W)
    sigma, u = split(w)
    tau, v = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)
    f = Function(DG).interpolate(-8 * pi * pi * cos(2 * pi * x + pi / 3) * cos(2 * pi * y))
    u1 = Function(DG).interpolate(cos(2 * pi * y) / 2)

    a = (dot(sigma, tau) + div(tau) * u + div(sigma) * v) * dx
    L = dot(tau, FacetNormal(mesh)) * u1 * ds(1) + f * v * dx

    g2 = Function(BDM).project(as_vector([-2 * pi * sqrt(3) / 2 * cos(2 * pi * y), -pi * sin(2 * pi * y)]))
    g3 = Function(BDM).project(as_vector([-2 * pi * sin(2 * pi * x + pi / 3), 0]))
    g4 = Function(BDM).project(as_vector([-2 * pi * sin(2 * pi * x + pi / 3), 0]))

    bc2 = EquationBC(dot(tau, sigma) * ds(2) - dot(tau, g2) * ds(2) == 0, w, 2, sub_space_index=0)
    bc3 = EquationBC(dot(tau, sigma) * ds(3) - dot(tau, g3) * ds(3) == 0, w, 3, sub_space_index=0)
    bc4 = DirichletBC(W.sub(0), g4, 4)

    solve(a - L == 0, w, bcs=[bc2, bc3, bc4], solver_parameters={'ksp_type': 'gmres', 'ksp_rtol': 1.e-10, 'ksp_atol': 1.e-10, 'ksp_max_it': 500000, 'mat_type': mat_type})

    f.interpolate(cos(2 * pi * x + pi / 3) * cos(2 * pi * y))
    g = Function(BDM).project(as_vector([-2 * pi * sin(2 * pi * x + pi / 3) * cos(2 * pi * y), -2 * pi * cos(2 * pi * x + pi / 3) * sin(2 * pi * y)]))

    return sqrt(assemble(dot(u - f, u - f) * dx)), sqrt(assemble(dot(sigma - g, sigma - g) * dx))


def linear_poisson_mixed(mat_type, mesh_num, porder):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    BDM = FunctionSpace(mesh, "BDM", porder+1)
    DG = FunctionSpace(mesh, "DG", porder)
    W = BDM * DG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)
    f = Function(DG).interpolate(-8 * pi * pi * cos(2 * pi * x + pi / 3) * cos(2 * pi * y))
    u1 = Function(DG).interpolate(cos(2 * pi * y) / 2)

    a = (dot(sigma, tau) + div(tau) * u + div(sigma) * v) * dx
    L = dot(tau, FacetNormal(mesh)) * u1 * ds(1) + f * v * dx

    g2 = Function(BDM).project(as_vector([-2 * pi * sqrt(3) / 2 * cos(2 * pi * y), -pi * sin(2 * pi * y)]))
    g3 = Function(BDM).project(as_vector([-2 * pi * sin(2 * pi * x + pi / 3), 0]))
    g4 = Function(BDM).project(as_vector([-2 * pi * sin(2 * pi * x + pi / 3), 0]))

    w = Function(W)

    bc2 = EquationBC(dot(tau, sigma) * ds(2) == dot(tau, g2) * ds(2), w, 2, sub_space_index=0)
    bc3 = EquationBC(dot(tau, sigma) * ds(3) == dot(tau, g3) * ds(3), w, 3, sub_space_index=0)
    bc4 = DirichletBC(W.sub(0), g4, 4)

    solve(a == L, w, bcs=[bc2, bc3, bc4], solver_parameters={'ksp_type': 'gmres', 'ksp_rtol': 1.e-10, 'ksp_atol': 1.e-10, 'ksp_max_it': 500000, 'mat_type': mat_type})

    sigma, u = w.split()

    f.interpolate(cos(2 * pi * x + pi / 3) * cos(2 * pi * y))
    g = Function(BDM).project(as_vector([-2 * pi * sin(2 * pi * x + pi / 3) * cos(2 * pi * y), -2 * pi * cos(2 * pi * x + pi / 3) * sin(2 * pi * y)]))

    return sqrt(assemble(dot(u - f, u - f) * dx)), sqrt(assemble(dot(sigma - g, sigma - g) * dx))


@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
@pytest.mark.parametrize("porder", [3])
def test_EquationBC_nonlinear_poisson(mat_type, porder):

    err = []
    for mesh_num in [8, 16]:
        err.append(nonlinear_poisson(mat_type, mesh_num, porder))

    assert(abs(math.log2(err[0]) - math.log2(err[1]) - (porder+1)) < 0.01)


@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
@pytest.mark.parametrize("porder", [3])
def test_EquationBC_linear_poisson(mat_type, porder):

    err = []
    for mesh_num in [8, 16]:
        err.append(linear_poisson(mat_type, mesh_num, porder))

    assert(abs(math.log2(err[0]) - math.log2(err[1]) - (porder+1)) < 0.01)


@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
@pytest.mark.parametrize("porder", [1])
def test_EquationBC_nonlinear_poisson_mixed(mat_type, porder):

    err = []
    for i, mesh_num in enumerate([8, 16]):
        err.append(nonlinear_poisson_mixed(mat_type, mesh_num, porder))

    assert(abs(math.log2(err[0][0]) - math.log2(err[1][0]) - (porder+1)) < 0.03)


@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
@pytest.mark.parametrize("porder", [1])
def test_EquationBC_linear_poisson_mixed(mat_type, porder):

    err = []
    for i, mesh_num in enumerate([8, 16]):
        err.append(linear_poisson_mixed(mat_type, mesh_num, porder))

    assert(abs(math.log2(err[0][0]) - math.log2(err[1][0]) - (porder+1)) < 0.03)
