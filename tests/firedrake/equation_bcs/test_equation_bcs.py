# Simple Poisson equation
# =========================

import pytest

from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER
import math


def nonlinear_poisson(solver_parameters, mesh_num, porder, pre_apply_bcs=True):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    V = FunctionSpace(mesh, "CG", porder)

    u = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = - 8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2)

    a = - inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    g = cos(2 * pi * x) * cos(2 * pi * y)

    # Equivalent to bc1 = EquationBC(v * (u - g1) * ds(1) == 0, u, 1)
    e2 = as_vector([0., 1.])
    bc1 = EquationBC((-inner(dot(grad(u), e2), dot(grad(v), e2)) + 3 * pi * pi * inner(u, v) + 1 * pi * pi * inner(g, v)) * ds(1) == 0, u, 1)

    solve(a - L == 0, u, bcs=[bc1], solver_parameters=solver_parameters, pre_apply_bcs=pre_apply_bcs)

    f = cos(x * pi * 2) * cos(y * pi * 2)
    return sqrt(assemble(inner(u - f, u - f) * dx))


def linear_poisson(solver_parameters, mesh_num, porder):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    V = FunctionSpace(mesh, "CG", porder)

    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = - 8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2)

    a = - inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    g = cos(2 * pi * x) * cos(2 * pi * y)

    u_ = Function(V)

    bc1 = EquationBC(inner(u, v) * ds(1) == inner(g, v) * ds(1), u_, 1)

    solve(a == L, u_, bcs=[bc1], solver_parameters=solver_parameters)

    f = cos(x * pi * 2) * cos(y * pi * 2)
    return sqrt(assemble(inner(u_ - f, u_ - f) * dx))


def nonlinear_poisson_bbc(solver_parameters, mesh_num, porder):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    V = FunctionSpace(mesh, "CG", porder)

    u = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = - 8.0 * pi * pi * cos(x * pi * 2)*cos(y * pi * 2)

    a = - inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    e2 = as_vector([0., 1.])
    a1 = (-inner(dot(grad(u), e2), dot(grad(v), e2)) + 4 * pi * pi * inner(u, v)) * ds(1)

    g = cos(2 * pi * x) * cos(2 * pi * y)
    bbc = DirichletBC(V, g, ((1, 3), (1, 4)))
    bc1 = EquationBC(a1 == 0, u, 1, bcs=[bbc])

    solve(a - L == 0, u, bcs=[bc1], solver_parameters=solver_parameters)

    f = cos(x * pi * 2) * cos(y * pi * 2)
    return sqrt(assemble(inner(u - f, u - f) * dx))


def linear_poisson_bbc(solver_parameters, mesh_num, porder):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    V = FunctionSpace(mesh, "CG", porder)

    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = - 8.0 * pi * pi * cos(x * pi * 2)*cos(y * pi * 2)

    a = - inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    e2 = as_vector([0., 1.])
    a1 = (-inner(dot(grad(u), e2), dot(e2, grad(v))) + 4 * pi * pi * inner(u, v)) * ds(1)
    L1 = inner(Constant(0), v) * ds(1)

    u = Function(V)

    g = cos(2 * pi * x) * cos(2 * pi * y)
    bbc = DirichletBC(V, g, ((1, 3), (1, 4)))
    bc1 = EquationBC(a1 == L1, u, 1, bcs=[bbc])

    solve(a == L, u, bcs=[bc1], solver_parameters=solver_parameters)

    f = cos(x * pi * 2)*cos(y * pi * 2)
    return sqrt(assemble(inner(u - f, u - f) * dx))


def nonlinear_poisson_mixed(solver_parameters, mesh_num, porder):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    BDM = FunctionSpace(mesh, "BDM", porder+1)
    DG = FunctionSpace(mesh, "DG", porder)
    W = BDM * DG

    w = Function(W)
    sigma, u = split(w)
    tau, v = TestFunctions(W)
    n = FacetNormal(mesh)

    x, y = SpatialCoordinate(mesh)
    f = -8 * pi * pi * cos(2 * pi * x + pi / 3) * cos(2 * pi * y)
    u1 = cos(2 * pi * y) / 2

    a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
    L = inner(u1, dot(tau, n)) * ds(1) + inner(f, v) * dx

    g = as_vector([-2 * pi * sin(2 * pi * x + pi / 3) * cos(2 * pi * y), -2 * pi * cos(2 * pi * x + pi / 3) * sin(2 * pi * y)])

    tau_n = dot(tau, n)
    sig_n = dot(sigma, n)
    g_n = dot(g, n)
    bc2 = EquationBC(inner(sig_n - g_n, tau_n) * ds(2) == 0, w, 2, V=W.sub(0))
    bc3 = EquationBC(inner(sig_n - g_n, tau_n) * ds(3) == 0, w, 3, V=W.sub(0))
    bc4 = DirichletBC(W.sub(0), g, 4)

    solve(a - L == 0, w, bcs=[bc2, bc3, bc4], solver_parameters=solver_parameters)

    f = cos(2 * pi * x + pi / 3) * cos(2 * pi * y)
    g = as_vector([-2 * pi * sin(2 * pi * x + pi / 3) * cos(2 * pi * y), -2 * pi * cos(2 * pi * x + pi / 3) * sin(2 * pi * y)])

    return sqrt(assemble(inner(u - f, u - f) * dx)), sqrt(assemble(inner(sigma - g, sigma - g) * dx))


def linear_poisson_mixed(solver_parameters, mesh_num, porder):

    mesh = UnitSquareMesh(mesh_num, mesh_num)

    BDM = FunctionSpace(mesh, "BDM", porder+1)
    DG = FunctionSpace(mesh, "DG", porder)
    W = BDM * DG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)
    f = -8 * pi * pi * cos(2 * pi * x + pi / 3) * cos(2 * pi * y)
    u1 = cos(2 * pi * y) / 2
    n = FacetNormal(mesh)

    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
    L = inner(u1, dot(tau, n)) * ds(1) + inner(f, v) * dx

    g = as_vector([-2 * pi * sin(2 * pi * x + pi / 3) * cos(2 * pi * y), -2 * pi * cos(2 * pi * x + pi / 3) * sin(2 * pi * y)])

    w = Function(W)

    tau_n = dot(tau, n)
    sig_n = dot(sigma, n)
    g_n = dot(g, n)
    bc2 = EquationBC(inner(sig_n, tau_n) * ds(2) == inner(g_n, tau_n) * ds(2), w, 2, V=W.sub(0))
    bc3 = EquationBC(inner(sig_n, tau_n) * ds(3) == inner(g_n, tau_n) * ds(3), w, 3, V=W.sub(0))
    bc4 = DirichletBC(W.sub(0), g, 4)

    solve(a == L, w, bcs=[bc2, bc3, bc4], solver_parameters=solver_parameters)

    f = cos(2 * pi * x + pi / 3) * cos(2 * pi * y)
    g = as_vector([-2 * pi * sin(2 * pi * x + pi / 3) * cos(2 * pi * y), -2 * pi * cos(2 * pi * x + pi / 3) * sin(2 * pi * y)])

    sigma, u = w.subfunctions
    return sqrt(assemble(inner(u - f, u - f) * dx)), sqrt(assemble(inner(sigma - g, sigma - g) * dx))


@pytest.mark.parametrize("eq_type", ["linear", "nonlinear"])
@pytest.mark.parametrize("with_bbc", [False, True])
@pytest.mark.parametrize("pre_apply_bcs", [False, True])
def test_EquationBC_poisson_matrix(eq_type, with_bbc, pre_apply_bcs):

    # Only test pre_apply_bcs=False for nonlinear case
    if not pre_apply_bcs and (eq_type == "linear"):
        pytest.skip(reason="Only test pre_apply_bcs=False in the nonlinear case")

    mat_type = "aij"
    porder = 2
    # Test standard poisson with EquationBCs
    # aij

    solver_parameters = {'mat_type': mat_type,
                         'ksp_type': 'preonly',
                         'pc_type': 'lu'}
    err = []
    mesh_sizes = [8, 16]
    if with_bbc:
        # test bcs for bcs
        if eq_type == "linear":
            for mesh_num in mesh_sizes:
                err.append(linear_poisson_bbc(solver_parameters, mesh_num, porder))
        elif eq_type == "nonlinear":
            for mesh_num in mesh_sizes:
                err.append(nonlinear_poisson_bbc(solver_parameters, mesh_num, porder))
    else:
        # test bcs for bcs
        if eq_type == "linear":
            for mesh_num in mesh_sizes:
                err.append(linear_poisson(solver_parameters, mesh_num, porder))
        elif eq_type == "nonlinear":
            for mesh_num in mesh_sizes:
                err.append(nonlinear_poisson(solver_parameters, mesh_num, porder, pre_apply_bcs=pre_apply_bcs))

    assert abs(math.log2(err[0]) - math.log2(err[1]) - (porder+1)) < 0.05


@pytest.mark.parametrize("with_bbc", [False, True])
def test_EquationBC_poisson_matfree(with_bbc):
    eq_type = "linear"
    mat_type = "matfree"
    porder = 2
    # Test standard poisson with EquationBCs
    # matfree

    solver_parameters = {'mat_type': mat_type,
                         'ksp_type': 'gmres',
                         'pc_type': 'none',
                         'ksp_atol': 1e-10,
                         'ksp_rtol': 1e-10,
                         'ksp_max_it': 200000,
                         'ksp_divtol': 1e8}
    err = []
    mesh_sizes = [8, 16]
    if with_bbc:
        if eq_type == "linear":
            for mesh_num in mesh_sizes:
                err.append(linear_poisson_bbc(solver_parameters, mesh_num, porder))
        elif eq_type == "nonlinear":
            for mesh_num in mesh_sizes:
                err.append(nonlinear_poisson_bbc(solver_parameters, mesh_num, porder))
    else:
        if eq_type == "linear":
            for mesh_num in mesh_sizes:
                err.append(linear_poisson(solver_parameters, mesh_num, porder))
        elif eq_type == "nonlinear":
            for mesh_num in mesh_sizes:
                err.append(nonlinear_poisson(solver_parameters, mesh_num, porder))

    assert abs(math.log2(err[0]) - math.log2(err[1]) - (porder+1)) < 0.05


# This test is so sensitive it will not pass unless MUMPS is used
@pytest.mark.skipmumps
@pytest.mark.parametrize("eq_type", ["linear", "nonlinear"])
def test_EquationBC_mixedpoisson_matrix(eq_type):
    mat_type = "aij"
    porder = 0
    # Mixed poisson with EquationBCs
    # aij

    solver_parameters = {"mat_type": mat_type,
                         "ksp_type": "preonly",
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}
    err = []
    mesh_sizes = [16, 32]
    if eq_type == "linear":
        for i, mesh_num in enumerate(mesh_sizes):
            err.append(linear_poisson_mixed(solver_parameters, mesh_num, porder))
    elif eq_type == "nonlinear":
        for i, mesh_num in enumerate(mesh_sizes):
            err.append(nonlinear_poisson_mixed(solver_parameters, mesh_num, porder))

    assert abs(math.log2(err[0][0]) - math.log2(err[1][0]) - (porder+1)) < 0.05


def test_EquationBC_mixedpoisson_matrix_fieldsplit():
    mat_type = "aij"
    eq_type = "linear"
    porder = 0
    # Mixed poisson with EquationBCs
    # aij with fieldsplit pc

    solver_parameters = {"mat_type": mat_type,
                         "ksp_type": "fgmres",
                         "ksp_rtol": 1.e-8,
                         "ksp_max_it": 200,
                         "pc_type": "fieldsplit",
                         "pc_fieldsplit_type": "schur",
                         "pc_fieldsplit_schur_fact_type": "full",
                         "fieldsplit_0_ksp_type": "preonly",
                         "fieldsplit_0_pc_type": "lu",
                         "fieldsplit_1_ksp_type": "cg",
                         "fieldsplit_1_pc_type": "none"}
    err = []
    mesh_sizes = [16, 32]
    if eq_type == "linear":
        for i, mesh_num in enumerate(mesh_sizes):
            err.append(linear_poisson_mixed(solver_parameters, mesh_num, porder))
    elif eq_type == "nonlinear":
        for i, mesh_num in enumerate(mesh_sizes):
            err.append(nonlinear_poisson_mixed(solver_parameters, mesh_num, porder))

    assert abs(math.log2(err[0][0]) - math.log2(err[1][0]) - (porder+1)) < 0.05


def test_EquationBC_mixedpoisson_matfree_fieldsplit():
    mat_type = "matfree"
    eq_type = "linear"
    porder = 0
    # Mixed poisson with EquationBCs
    # matfree with fieldsplit pc

    solver_parameters = {'mat_type': mat_type,
                         'ksp_type': 'fgmres',
                         'ksp_atol': 1e-11,
                         'ksp_max_it': 200,
                         'pc_type': 'fieldsplit',
                         'pc_fieldsplit_type': 'schur',
                         'pc_fieldsplit_schur_fact_type': 'full',
                         'fieldsplit_0_ksp_type': 'cg',
                         'fieldsplit_0_pc_type': 'python',
                         'fieldsplit_0_pc_python_type': 'firedrake.AssembledPC',
                         'fieldsplit_0_assembled_pc_type': 'lu',
                         'fieldsplit_1_ksp_type': 'cg',
                         'fieldsplit_1_pc_use_amat': True,
                         'fieldsplit_1_pc_type': 'python',
                         'fieldsplit_1_pc_python_type': 'firedrake.MassInvPC',
                         'fieldsplit_1_Mp_pc_type': 'icc'}
    err = []
    mesh_sizes = [16, 32]
    if eq_type == "linear":
        for i, mesh_num in enumerate(mesh_sizes):
            err.append(linear_poisson_mixed(solver_parameters, mesh_num, porder))
    elif eq_type == "nonlinear":
        for i, mesh_num in enumerate(mesh_sizes):
            err.append(nonlinear_poisson_mixed(solver_parameters, mesh_num, porder))

    assert abs(math.log2(err[0][0]) - math.log2(err[1][0]) - (porder+1)) < 0.05


def test_equation_bcs_pc():
    mesh = UnitSquareMesh(2**6, 2**6)
    CG = FunctionSpace(mesh, "CG", 3)
    R = FunctionSpace(mesh, "R", 0)
    V = CG * R
    f = Function(V)
    u, l = split(f)
    v, w = split(TestFunction(V))
    x, y = SpatialCoordinate(mesh)
    exact = cos(2 * pi * x) * cos(2 * pi * y)
    g = 8 * pi**2 * exact
    F = inner(grad(u), grad(v)) * dx + inner(l, w) * dx - inner(g, v) * dx
    bc = EquationBC(inner((u - exact), v) * ds == 0, f, (1, 2, 3, 4), V=V.sub(0))
    params = {
        "mat_type": "matfree",
        "ksp_type": "fgmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_0_fields": "0",
        "pc_fieldsplit_1_fields": "1",
        "fieldsplit_0": {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled": {
                "ksp_type": "cg",
                "pc_type": "asm",
            },
        },
        "fieldsplit_1": {
            "ksp_type": "gmres",
            "max_it": 1,
            "convergence_test": "skip",
        }
    }
    problem = NonlinearVariationalProblem(F, f, bcs=[bc])
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    error = assemble(inner(u - exact, u - exact) * dx)**0.5
    assert error < 1.e-7
