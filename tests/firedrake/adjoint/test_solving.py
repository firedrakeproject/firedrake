import pytest

from firedrake import *
from firedrake.adjoint import *
from numpy.testing import assert_approx_equal


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.fixture
def rg():
    return RandomGenerator(PCG64(seed=1234))


@pytest.mark.skipcomplex
def test_linear_problem(rg):
    assert len(get_working_tape()._blocks) == 0
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    R = FunctionSpace(mesh, "R", 0)
    f = Function(V).assign(1.)

    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1), "on_boundary")

    def J(f):
        a = inner(grad(u), grad(v))*dx
        L = f*v*dx
        solve(a == L, u_, bcs=bc)
        return assemble(u_**2*dx)

    # this tests a bug in recompute() if bcs= keyword argument is provided in solve
    J0 = J(f)
    rf = ReducedFunctional(J0, Control(f))
    assert_approx_equal(rf(f), J0)
    assert rf.tape.recompute_count == 1
    _test_adjoint(J, f, rg)


@pytest.mark.skipcomplex
def test_singular_linear_problem(rg):
    """This tests whether nullspace and solver_parameters are passed on in adjoint solves"""
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V).assign(1.)

    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)
    nullspace = VectorSpaceBasis(constant=True)
    solver_parameters = {'ksp_type': 'cg', 'pc_type': 'sor'}

    def J(f):
        a = inner(grad(u), grad(v))*dx
        L = f*v*dx
        solve(a == L, u_, nullspace=nullspace, transpose_nullspace=nullspace,
              solver_parameters=solver_parameters)
        return assemble(u_**2*dx)

    _test_adjoint(J, f, rg)


@pytest.mark.skipcomplex
@pytest.mark.parametrize("pre_apply_bcs", (True, False))
def test_nonlinear_problem(pre_apply_bcs, rg):
    """This tests whether nullspace and solver_parameters are passed on in adjoint solves"""
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    R = FunctionSpace(mesh, "R", 0)
    f = Function(V).assign(1.)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1), "on_boundary")

    def J(f):
        a = f*inner(grad(u), grad(v))*dx + u**2*v*dx - f*v*dx
        L = 0
        solve(a == L, u, bc, pre_apply_bcs=pre_apply_bcs)
        return assemble(u**2*dx)

    _test_adjoint(J, f, rg)


@pytest.mark.skipcomplex
def test_mixed_boundary(rg):
    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc1 = DirichletBC(V, y*y, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1, bc2]

    g1 = Constant(2)
    g2 = Constant(1)
    f = Function(V).assign(10.)

    def J(f):
        a = f*inner(grad(u), grad(v))*dx
        L = inner(f, v)*dx + inner(g1, v)*ds(4) + inner(g2, v)*ds(3)

        solve(a == L, u_, bc)

        return assemble(u_**2*dx)

    _test_adjoint(J, f, rg)


def xtest_wrt_constant_dirichlet_boundary():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "Lagrange", 1)

    c = Constant(1)

    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    def J(bc):
        a = inner(grad(u), grad(v))*dx
        L = c*v*dx
        solve(a == L, u_, bc)
        return assemble(u_**2*dx)

    _test_adjoint_constant_boundary(J, bc)


def xtest_wrt_function_dirichlet_boundary():
    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc_func = project(sin(y), V)
    bc1 = DirichletBC(V, bc_func, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1, bc2]

    g1 = Constant(2)
    g2 = Constant(1)
    f = Function(V).assign(10.)

    def J(bc):
        a = inner(grad(u), grad(v))*dx
        L = inner(f, v)*dx + inner(g1, v)*ds(4) + inner(g2, v)*ds(3)

        solve(a == L, u_, [bc, bc2])

        return assemble(u_**2*dx)

    _test_adjoint_function_boundary(J(bc), bc1, bc_func)


@pytest.mark.skipcomplex
def test_wrt_function_neumann_boundary():
    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc1 = DirichletBC(V, y*y, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1, bc2]

    g1 = Function(R, val=2)
    g2 = Function(R, val=1)
    f = Function(V).assign(10.)

    def J(g1):
        a = inner(grad(u), grad(v))*dx
        L = inner(f, v)*dx + inner(g1, v)*ds(4) + inner(g2, v)*ds(3)

        solve(a == L, u_, bc)

        return assemble(u_**2*dx)

    _test_adjoint_constant(J, g1)


@pytest.mark.skipcomplex
def test_wrt_constant():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    R = FunctionSpace(mesh, "R", 0)

    c = Function(R, val=1)

    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1), "on_boundary")

    def J(c):
        a = inner(grad(u), grad(v))*dx
        L = c*v*dx
        solve(a == L, u_, bc)
        return assemble(u_**2*dx)

    _test_adjoint_constant(J, c)


@pytest.mark.skipcomplex
def test_wrt_constant_neumann_boundary():
    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc1 = DirichletBC(V, y*y, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1, bc2]

    g1 = Function(R, val=2)
    g2 = Function(R, val=1)
    f = Function(V).assign(10.)

    def J(g1):
        a = inner(grad(u), grad(v))*dx
        L = inner(f, v)*dx + inner(g1, v)*ds(4) + inner(g2, v)*ds(3)

        solve(a == L, u_, bc)

        return assemble(u_**2*dx)

    _test_adjoint_constant(J, g1)


@pytest.mark.skipcomplex
def test_time_dependent():
    # Defining the domain, 100 points from 0 to 1
    mesh = IntervalMesh(100, 0, 1)

    # Defining function space, test and trial functions
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    # Dirichlet boundary conditions
    bc_left = DirichletBC(V, 1, 1)
    bc_right = DirichletBC(V, 2, 2)
    bc = [bc_left, bc_right]

    # Some variables
    T = 0.2
    dt = 0.1
    f = Function(R, val=1)

    def J(f):
        u_1 = Function(V).assign(1.)

        a = u_1*u*v*dx + dt*f*inner(grad(u), grad(v))*dx
        L = u_1*v*dx

        # Time loop
        t = dt
        while t <= T:
            solve(a == L, u_, bc)
            u_1.assign(u_)
            t += dt

        return assemble(u_1**2*dx)

    _test_adjoint_constant(J, f)


@pytest.mark.skipcomplex
def test_two_nonlinear_solves():
    # regression test for firedrake issue #1841
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    v = TestFunction(V)
    u0 = Function(V)
    u1 = Function(V)

    ui = Function(R, val=2.0)
    c = Control(ui)
    u0.assign(ui)
    F = dot(v, (u1-u0))*dx - dot(v, u0*u1)*dx
    problem = NonlinearVariationalProblem(F, u1)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()
    u0.assign(u1)
    solver.solve()
    J = assemble(dot(u1, u1)*dx)
    rf = ReducedFunctional(J, c)
    assert taylor_test(rf, ui, Constant(0.1)) > 1.95
    # Taylor test recomputes the functional 5 times.
    assert rf.tape.recompute_count == 5


@pytest.mark.skipcomplex
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.usefixtures("garbage_cleanup")
@pytest.mark.parametrize("constant_jacobian", [False, True])
def test_cached_adjoint_rhs(constant_jacobian: bool) -> None:
    """A cached adjoint solver uses the current RHS for every derivative.

    Interpolation adds a separate term to the adjoint residual. Since the
    interpolation is the identity on V, the forward solution is exactly f/2.
    """
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)
    f = Function(V).interpolate(1 + x)
    u = Function(V)
    v = TestFunction(V)
    F = inner(interpolate(u, V), v)*dx + inner(u - f, v)*dx
    problem = NonlinearVariationalProblem(F, u)
    problem._constant_jacobian = constant_jacobian
    solver = NonlinearVariationalSolver(problem, solver_parameters={
        "ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12,
    })
    solver.solve()
    linear = ReducedFunctional(assemble(u*dx), Control(f))
    quadratic = ReducedFunctional(assemble(0.5*u**2*dx), Control(f))
    adjoint_solver = solver._ad_solvers["adjoint_lvs"]

    for functional, seed, density in (
        (linear, 1.0, 0.5),
        (quadratic, -2.0, f/4),
        (quadratic, 0.0, f/4),
        (linear, 3.0, 0.5),
    ):
        gradient = functional.derivative(adj_input=seed)
        with stop_annotating():
            expected = assemble(Constant(seed)*density*v*dx)
            error = assemble(gradient - expected)
        with error.dat.vec_ro as vec:
            assert vec.norm() < 1e-11
        assert solver._ad_solvers["adjoint_lvs"] is adjoint_solver


@pytest.mark.skipcomplex
def test_real_solve(rg):
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    u = TrialFunction(R)
    v = TestFunction(R)
    m = Function(V).assign(1.0)

    def J(m):
        a = u * v * dx
        L = m * v * dx
        solution = Function(R)
        solve(a == L, solution)
        return assemble(solution**2 * dx)

    _test_adjoint(J, m, rg)


@pytest.mark.skipcomplex
def test_multiple_meshes(rg):
    mesh1 = UnitSquareMesh(4, 4)
    mesh2 = RectangleMesh(nx=4, ny=4, Lx=3, Ly=1, originX=2, originY=0)

    V1 = FunctionSpace(mesh1, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)
    V = V1*V2

    u = Function(V)
    u1, u2 = split(u)
    v1, v2 = TestFunctions(V)

    f = Function(V).assign(10.)
    f1, f2 = split(f)

    a = inner(grad(u1), grad(v1))*dx(mesh1) + inner(grad(u2), grad(v2))*dx(mesh2)
    L = inner(f1, v1)*dx(mesh1) + inner(f2, v2)*dx(mesh2)

    bc1 = DirichletBC(V.sub(0), 0, "on_boundary")
    bc2 = DirichletBC(V.sub(1), 0, "on_boundary")
    bcs = [bc1, bc2]

    solve(a - L == 0, u, bcs)

    J = assemble(u1**4*dx(mesh1) + u2**4*dx(mesh2))
    rf = ReducedFunctional(J, Control(f))
    df = rg.uniform(V)

    taylor = taylor_to_dict(rf, f, df)

    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']


@pytest.mark.skipcomplex
def test_submesh(rg):
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)

    DG = FunctionSpace(mesh, "DG", 0)
    ind = Function(DG).interpolate(conditional(y > 0.5, 1, 0))
    relabeled_mesh = RelabeledMesh(mesh, [ind], [10])
    submesh = Submesh(relabeled_mesh, 2, 10)
    dx_sub = Measure("dx", domain=submesh, intersect_measures=(Measure("dx", relabeled_mesh),))

    V1 = FunctionSpace(relabeled_mesh, "CG", 1)
    V2 = FunctionSpace(submesh, "CG", 1)
    V = V1*V2

    u = Function(V)
    u1, u2 = split(u)
    v1, v2 = TestFunctions(V)

    f = Function(V1).assign(10.)

    a = inner(grad(u1), grad(v1))*dx(relabeled_mesh)
    a += inner(u1 - u2, v2)*dx_sub
    L = inner(f, v1)*dx(relabeled_mesh)

    bcs = [DirichletBC(V.sub(0), 0, "on_boundary")]

    solve(a - L == 0, u, bcs)

    J = assemble(u2**4*dx_sub)
    rf = ReducedFunctional(J, Control(f))
    df = rg.uniform(V1)

    taylor = taylor_to_dict(rf, f, df)

    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']


def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))

    return r


def _test_adjoint_function_boundary(J, bc, f):
    tape = Tape()
    set_working_tape(tape)

    V = f.function_space()
    h = Function(V).assign(1.)
    g = Function(V)
    eps_ = [0.4/2.0**i for i in range(4)]
    residuals = []
    for eps in eps_:
        g.assign(f + eps*h)
        bc.set_value(g)
        Jp = J(bc)
        tape.clear_tape()
        bc.set_value(f)
        Jm = J(bc)
        Jm.block_variable.adj_value = 1.0
        tape.evaluate_adj()

        dJdbc = bc.block_variable.adj_value

        residual = abs(Jp - Jm - eps*dJdbc.inner(h))
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)

    tol = 1E-1
    assert (r[-1] > 2-tol)


def _test_adjoint_constant_boundary(J, bc):
    tape = Tape()
    set_working_tape(tape)

    h = Constant(1)
    c = Constant(1)

    eps_ = [0.4/2.0**i for i in range(4)]
    residuals = []
    for eps in eps_:
        bc.set_value(Constant(c + eps*h))
        Jp = J(bc)
        tape.clear_tape()
        bc.set_value(c)
        Jm = J(bc)
        Jm.block_variable.adj_value = 1.0
        tape.evaluate_adj()

        dJdbc = bc.block_variable.adj_value[0]

        residual = abs(Jp - Jm - eps*dJdbc.sum())
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)

    tol = 1E-1
    assert (r[-1] > 2-tol)


def _test_adjoint_constant(J, c):
    tape = Tape()
    set_working_tape(tape)

    h = Constant(1)

    eps_ = [0.01/2.0**i for i in range(4)]
    residuals = []
    for eps in eps_:

        Jp = J(c + eps*h)
        tape.clear_tape()
        Jm = J(c)
        Jm.block_variable.adj_value = 1.0
        tape.evaluate_adj()

        dJdc = c.block_variable.adj_value.dat.data_ro[0]

        residual = abs(Jp - Jm - eps*dJdc)
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)

    tol = 1E-1
    assert (r[-1] > 2-tol)


def _test_adjoint(J, f, rg):
    tape = Tape()
    set_working_tape(tape)

    V = f.function_space()
    h = rg.uniform(V)

    eps_ = [0.01/2.0**i for i in range(5)]
    residuals = []
    for eps in eps_:

        Jp = J(f + eps*h)
        tape.clear_tape()
        Jm = J(f)
        Jm.block_variable.adj_value = 1.0
        tape.evaluate_adj()

        dJdf = f.block_variable.adj_value.dat

        residual = abs(Jp - Jm - eps*dJdf.inner(h.dat))
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)

    tol = 1E-1
    assert (r[-1] > 2-tol)
