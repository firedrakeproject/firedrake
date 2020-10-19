from firedrake import *
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

    solve(a == L, w, bcs=bcs, nullspace=nullspace)

    exact = Function(DG)
    exact.interpolate(x[1] - 0.5)

    sigma, u = w.split()
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

    sigma, u = w.split()
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
    n_interp = [interpolate(n, V) for n in ns]
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


def test_near_nullspace_mixed():
    # test nullspace and nearnullspace for a mixed Stokes system
    # this is tested on the SOLCX case of May and Moresi https://doi.org/10.1016/j.pepi.2008.07.036
    PETSc.Sys.popErrorHandler()
    n = 256
    mesh = UnitSquareMesh(n, n)
    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)
    W = V*P

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    x, y = SpatialCoordinate(mesh)
    inside_box = And(abs(x-0.5)<0.2, abs(y-0.75)<0.2)
    mu = conditional(x<0.5, 1, 1e8)  # SOLCX
    #mu = conditional(inside_box, 1e8, 1) # SINKER
    #mu = 1e6*exp(y) # SOLKY
    a = inner(mu*grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx

    f = as_vector((0, sin(pi*y)*cos(pi*x))) # SOLCX/SINKER
    #f = as_vector((0, -9.8*conditional(inside_box, 2, 1))) # SOLKY
    L = inner(f, v)*dx

    bcs = [DirichletBC(W[0].sub(0), 0, (1, 2)), DirichletBC(W[0].sub(1), 0, (3, 4))]
    bcs = [DirichletBC(W[0], Constant((0., 0.)), (1,2,3,4))]

    rotW = Function(W)
    rotV, _ = rotW.split()
    rotV.interpolate(as_vector((-y, x)))

    c0 = Function(W)
    c0V, _ = c0.split()
    c1 = Function(W)
    c1V, _ = c1.split()
    c0V.interpolate(Constant([1., 0.]))
    c1V.interpolate(Constant([0., 1.]))

    near_nullmodes = VectorSpaceBasis([c0V, c1V, rotV])
    near_nullmodes.orthonormalize()
    near_nullmodes_W = MixedVectorSpaceBasis(W, [near_nullmodes, W.sub(1)])

    w = Function(W)
    u, p = w.split()

    solve(a == L, w, bcs=bcs, appctx={'mu': mu}, near_nullspace=near_nullmodes_W,
          solver_parameters={
        'mat_type': 'matfree',
        'pc_type': 'fieldsplit',
        'ksp_type': 'preonly',
        #'ksp_view': None,
        'pc_fieldsplit_type': 'schur',
        'fieldsplit_schur_fact_type': 'full',
        'fieldsplit_0': {
            'ksp_type': 'cg',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'gamg',
            'assembled_pc_gamg_threshold': 0.01,
            'assembled_pc_mg_distinct_smoothup': None,
            'assembled_mg_levels_ksp_type': 'richardson',
            'assembled_mg_levels_ksp_max_it': 1,
            'assembled_mg_levels_pc_type': 'sor',
            'assembled_mg_levels_pc_sor_local_forward': None,
            'assembled_mg_levels_up_ksp_type': 'richardson',
            'assembled_mg_levels_up_ksp_max_it': 1,
            'assembled_mg_levels_up_pc_type': 'sor',
            'assembled_mg_levels_up_pc_sor_local_backward': None,
            'ksp_rtol': '1e-9',
            'ksp_converged_reason': None,
        },
        'fieldsplit_1': {
            'ksp_type': 'fgmres',
            'ksp_converged_reason': None,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.MassInvPC',
            'Mp_ksp_type': 'cg',
            'Mp_pc_type': 'sor',
            'ksp_rtol': '1e-5',
            'ksp_monitor': None,
        }
    })

    File('u.pvd').write(u)
