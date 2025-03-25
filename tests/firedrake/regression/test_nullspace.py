from firedrake import *
from firedrake.__future__ import *
from firedrake.petsc import PETSc
import pytest
import numpy as np


@pytest.fixture(scope='module', params=[False, True])
def V(request):
    quadrilateral = request.param
    m = UnitSquareMesh(25, 25, quadrilateral=quadrilateral)
    return FunctionSpace(m, 'CG', 1)


def test_nullspace(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(V.mesh())

    a = inner(grad(u), grad(v))*dx
    L = -conj(v)*ds(3) + conj(v)*ds(4)

    nullspace = VectorSpaceBasis(constant=True)
    u = Function(V)
    solve(a == L, u, nullspace=nullspace)

    exact = Function(V)
    exact.interpolate(x[1] - 0.5)
    assert sqrt(assemble(inner((u - exact), (u - exact))*dx)) < 5e-8


def test_orthonormalize():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 1)
    a = Function(V).interpolate(Constant((2, 0)))
    b = Function(V).interpolate(Constant((0, 2)))

    basis = VectorSpaceBasis([a, b])
    assert basis.is_orthogonal()
    assert not basis.is_orthonormal()

    basis.orthonormalize()
    assert basis.is_orthogonal()
    assert basis.is_orthonormal()


def test_transpose_nullspace():
    errors = []
    for n in range(4, 10):
        mesh = UnitIntervalMesh(2**n)
        V = FunctionSpace(mesh, "CG", 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(grad(u), grad(v))*dx
        L = conj(v)*dx

        nullspace = VectorSpaceBasis(constant=True)
        u = Function(V)
        u.interpolate(SpatialCoordinate(mesh)[0])
        # Solver diverges with indefinite PC if we don't remove
        # transpose nullspace.
        solve(a == L, u, nullspace=nullspace,
              transpose_nullspace=nullspace,
              solver_parameters={"ksp_type": "cg",
                                 "ksp_initial_guess_non_zero": True,
                                 "pc_type": "gamg"})
        # Solution should integrate to 0.5
        errors.append(assemble(u*dx) - 0.5)
    errors = np.asarray(errors)
    rate = np.log2(errors[:-1] / errors[1:])
    assert (rate > 1.9).all()


def test_nullspace_preassembled(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(V.mesh())

    a = inner(grad(u), grad(v))*dx
    L = -conj(v)*ds(3) + conj(v)*ds(4)

    nullspace = VectorSpaceBasis(constant=True)
    u = Function(V)
    A = assemble(a)
    b = assemble(L)
    solve(A, u, b, nullspace=nullspace)

    exact = Function(V)
    exact.interpolate(x[1] - 0.5)
    assert sqrt(assemble(inner((u - exact), (u - exact))*dx)) < 5e-8


def test_nullspace_mixed():
    m = UnitSquareMesh(5, 5)
    x = SpatialCoordinate(m)
    BDM = FunctionSpace(m, 'BDM', 1)
    DG = FunctionSpace(m, 'DG', 0)
    W = BDM * DG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v))*dx

    bc1 = Function(BDM).assign(0.0)
    bc2 = Function(BDM).project(Constant((0, 1)))

    bcs = [DirichletBC(W.sub(0), bc1, (1, 2)),
           DirichletBC(W.sub(0), bc2, (3, 4))]

    w = Function(W)

    f = Function(DG)
    f.assign(0)
    L = inner(f, v)*dx

    # Null space is constant functions in DG and empty in BDM.
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

    solve(a == L, w, bcs=bcs, nullspace=nullspace,
          solver_parameters={
              'ksp_type': 'minres',
              'ksp_converged_reason': None,
              'pc_type': 'none'})

    exact = Function(DG)
    exact.interpolate(x[1] - 0.5)

    sigma, u = w.subfunctions
    assert sqrt(assemble(inner((u - exact), (u - exact))*dx)) < 1e-7

    # Now using a Schur complement
    w.assign(0)
    solve(a == L, w, bcs=bcs, nullspace=nullspace,
          solver_parameters={'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'schur',
                             'ksp_type': 'cg',
                             'pc_fieldsplit_schur_fact_type': 'full',
                             'fieldsplit_0_ksp_type': 'preonly',
                             'fieldsplit_0_pc_type': 'lu',
                             'fieldsplit_1_ksp_type': 'cg',
                             'fieldsplit_1_pc_type': 'none'})

    sigma, u = w.subfunctions
    assert sqrt(assemble(inner((u - exact), (u - exact))*dx)) < 5e-8


def test_near_nullspace(tmpdir):
    # Tests the near nullspace for the case of the linear elasticity equations
    mesh = UnitSquareMesh(100, 100)
    x, y = SpatialCoordinate(mesh)
    dim = 2
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    mu = Constant(0.2)
    lmbda = Constant(0.3)

    def sigma(fn):
        return 2.0 * mu * sym(grad(fn)) + lmbda * tr(sym(grad(fn))) * Identity(dim)

    w_exact = Function(V)
    w_exact.interpolate(as_vector([x*y, x*y]))
    f = Constant((mu + lmbda, mu + lmbda))
    F = inner(sigma(u), grad(v))*dx + inner(f, v)*dx

    bcs = [DirichletBC(V, w_exact, (1, 2, 3, 4))]

    n0 = Constant((1, 0))
    n1 = Constant((0, 1))
    n2 = as_vector([y - 0.5, -(x - 0.5)])
    ns = [n0, n1, n2]
    n_interp = [assemble(interpolate(n, V)) for n in ns]
    nsp = VectorSpaceBasis(vecs=n_interp)
    nsp.orthonormalize()

    wo_nns_log = str(tmpdir.join("wo_nns_log"))
    w_nns_log = str(tmpdir.join("w_nns_log"))

    w1 = Function(V)
    solve(lhs(F) == rhs(F), w1, bcs=bcs, solver_parameters={
        'ksp_monitor_short': "ascii:%s:" % w_nns_log,
        'ksp_rtol': 1e-8, 'ksp_atol': 1e-8, 'ksp_type': 'cg',
        'pc_type': 'gamg',
        'mg_levels_ksp_max_it': 3,
        'mat_type': 'aij'}, near_nullspace=nsp)

    w2 = Function(V)
    solve(lhs(F) == rhs(F), w2, bcs=bcs, solver_parameters={
        'ksp_monitor_short': "ascii:%s:" % wo_nns_log,
        'ksp_rtol': 1e-8, 'ksp_atol': 1e-8, 'ksp_type': 'cg',
        'pc_type': 'gamg',
        'mg_levels_ksp_max_it': 3,
        'mat_type': 'aij'})

    # check that both solutions are equal to the exact solution
    assert sqrt(assemble(inner(w1-w2, w1-w2)*dx)) < 1e-7
    assert sqrt(assemble(inner(w1-w_exact, w1-w_exact)*dx)) < 1e-7

    with open(wo_nns_log, "r") as f:
        f.readline()
        wo = f.read()

    with open(w_nns_log, "r") as f:
        f.readline()
        w = f.read()

    # Check that the number of iterations necessary decreases when using near nullspace
    assert (len(w.split("\n"))-1) <= 0.75 * (len(wo.split("\n"))-1)


def test_nullspace_mixed_multiple_components():
    # tests mixed nullspace with nullspace components in both spaces
    # and passing of sub-nullspace in fieldsplit

    mesh = PeriodicRectangleMesh(10, 10, 1, 1)

    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = MixedFunctionSpace([V, W])

    N, M = TestFunctions(Z)
    z = Function(Z)
    u, p = split(z)

    x, y = SpatialCoordinate(mesh)
    khat = Constant((0., -1.))
    mu = Constant(1.0)
    g = Constant(1.0)
    tau = mu * (grad(u)+transpose(grad(u)))
    # added constant to create inconsistent rhs:
    rho = sin(pi*y)*cos(pi*x) + Constant(1.0)

    F_stokes = inner(tau, grad(N)) * dx + inner(grad(p), N) * dx
    F_stokes += inner(g * rho * khat, N) * dx
    F_stokes += -inner(u, grad(M)) * dx

    PETSc.Sys.popErrorHandler()
    solver_parameters = {
        'mat_type': 'matfree',
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_type': 'full',
        'fieldsplit_0': {
            'ksp_type': 'cg',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'gamg',
            'ksp_rtol': '1e-9',
            'ksp_test_null_space': None,
            'ksp_converged_reason': None,
        },
        'fieldsplit_1': {
            'ksp_type': 'fgmres',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.MassInvPC',
            'Mp_pc_type': 'ksp',
            'Mp_ksp_ksp_type': 'cg',
            'Mp_ksp_pc_type': 'sor',
            'ksp_rtol': '1e-7',
            'ksp_monitor': None,
        }
    }

    ux0 = Function(Z.sub(0))
    ux0.assign(Constant((1.0, 0.)))
    uy0 = Function(Z.sub(0))
    uy0.assign(Constant((0.0, 1.)))
    uv_nullspace = VectorSpaceBasis([ux0, uy0])
    uv_nullspace.orthonormalize()

    p_nullspace = VectorSpaceBasis(constant=True)

    mix_nullspace = MixedVectorSpaceBasis(Z, [uv_nullspace, p_nullspace])

    problem = NonlinearVariationalProblem(F_stokes, z)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=solver_parameters,
        nullspace=mix_nullspace, transpose_nullspace=mix_nullspace)
    solver.solve()

    assert solver.snes.getConvergedReason() > 0
    schur_ksp = solver.snes.ksp.pc.getFieldSplitSubKSP()[1]
    assert schur_ksp.getConvergedReason() > 0
    assert schur_ksp.getIterationNumber() < 6


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("aux_pc", [False, True], ids=["PC(mu)", "PC(DG0-mu)"])
@pytest.mark.parametrize("rhs", ["form_rhs", "cofunc_rhs"])
def test_near_nullspace_mixed(aux_pc, rhs):
    # test nullspace and nearnullspace for a mixed Stokes system
    # this is tested on the SINKER case of May and Moresi https://doi.org/10.1016/j.pepi.2008.07.036
    # fails in parallel if nullspace is copied to fieldsplit_1_Mp_ksp solve (see PR #3488)
    PETSc.Sys.popErrorHandler()
    n = 64
    mesh = UnitSquareMesh(n, n)
    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)
    W = V*P

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)
    inside_box = And(abs(x-0.5) < 0.2, abs(y-0.75) < 0.2)
    mu = conditional(inside_box, 1e8, 1)

    # mu might vary within cells lying on the box boundary
    # we need a higher quadrature degree
    a = inner(mu*2*sym(grad(u)), grad(v))*dx(degree=6)
    a += -inner(p, div(v))*dx + inner(div(u), q)*dx
    aP = None
    mu0 = mu
    if aux_pc:
        DG0 = FunctionSpace(mesh, "DG", 0)
        mu0 = Function(DG0).interpolate(mu)
        aP = inner(mu0*2*sym(grad(u)), grad(v))*dx(degree=2)
        aP += -inner(p, div(v))*dx + inner(div(u), q)*dx

    f = as_vector((0, -9.8*conditional(inside_box, 2, 1)))
    L = inner(f, v)*dx
    if rhs == 'cofunc_rhs':
        L = assemble(L)
    elif rhs != 'form_rhs':
        raise ValueError("Unknown right hand side type")

    bcs = [DirichletBC(W[0].sub(0), 0, (1, 2)), DirichletBC(W[0].sub(1), 0, (3, 4))]

    rotW = Function(W)
    rotV, _ = rotW.subfunctions
    rotV.interpolate(as_vector((-y, x)))

    c0 = Function(W)
    c0V, _ = c0.subfunctions
    c1 = Function(W)
    c1V, _ = c1.subfunctions
    c0V.interpolate(Constant([1., 0.]))
    c1V.interpolate(Constant([0., 1.]))

    near_nullmodes = VectorSpaceBasis([c0V, c1V, rotV])
    near_nullmodes.orthonormalize()
    near_nullmodes_W = MixedVectorSpaceBasis(W, [near_nullmodes, W.sub(1)])

    pressure_nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

    w = Function(W)
    solver_parameters = {
        'mat_type': 'matfree',
        'pc_type': 'fieldsplit',
        'ksp_type': 'preonly',
        'pc_fieldsplit_type': 'schur',
        'fieldsplit_schur_fact_type': 'full',
        'fieldsplit_0': {
            'ksp_type': 'cg',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'gamg',
            'assembled_mg_levels_pc_type': 'sor',
            'assembled_mg_levels_pc_sor_diagonal_shift': 1e-100,  # See https://gitlab.com/petsc/petsc/-/issues/1221
            'ksp_rtol': 1e-7,
            'ksp_converged_reason': None,
        },
        'fieldsplit_1': {
            'ksp_type': 'fgmres',
            'ksp_converged_reason': None,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.MassInvPC',
            'Mp_pc_type': 'ksp',
            'Mp_ksp_ksp_type': 'cg',
            'Mp_ksp_pc_type': 'sor',
            'ksp_rtol': '1e-5',
            'ksp_monitor': None,
        }
    }

    problem = LinearVariationalProblem(a, L, w, bcs=bcs, aP=aP)
    solver = LinearVariationalSolver(
        problem, appctx={'mu': mu0},
        nullspace=pressure_nullspace,
        transpose_nullspace=pressure_nullspace,
        near_nullspace=near_nullmodes_W,
        solver_parameters=solver_parameters)
    solver.solve()

    assert solver.snes.getConvergedReason() > 0
    ksp_inner, _ = solver.snes.ksp.pc.getFieldSplitSubKSP()
    assert ksp_inner.getConvergedReason() > 0
    A, P = ksp_inner.getOperators()
    assert A.getNearNullSpace().handle
    # currently ~22 (25 on 2 cores) vs. >45-ish for with/without near nullspace
    assert ksp_inner.getIterationNumber() < 27
