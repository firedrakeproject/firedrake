import pytest

from firedrake import *
from firedrake.adjoint import *
from numpy.testing import assert_approx_equal


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    if annotate_tape():
        pause_annotation()


@pytest.mark.skipcomplex
def test_linear_problem():
    assert len(get_working_tape()._blocks) == 0
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    R = FunctionSpace(mesh, "R", 0)
    f = Function(V)
    f.vector()[:] = 1

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
    _test_adjoint(J, f)


@pytest.mark.skipcomplex
def test_singular_linear_problem():
    """This tests whether nullspace and solver_parameters are passed on in adjoint solves"""
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    f.vector()[:] = 1

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

    _test_adjoint(J, f)


@pytest.mark.skipcomplex
def test_nonlinear_problem():
    """This tests whether nullspace and solver_parameters are passed on in adjoint solves"""
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    R = FunctionSpace(mesh, "R", 0)
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1), "on_boundary")

    def J(f):
        a = f*inner(grad(u), grad(v))*dx + u**2*v*dx - f*v*dx
        L = 0
        solve(a == L, u, bc)
        return assemble(u**2*dx)

    _test_adjoint(J, f)


@pytest.mark.skipcomplex
def test_mixed_boundary():
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
    f = Function(V)
    f.vector()[:] = 10

    def J(f):
        a = f*inner(grad(u), grad(v))*dx
        L = inner(f, v)*dx + inner(g1, v)*ds(4) + inner(g2, v)*ds(3)

        solve(a == L, u_, bc)

        return assemble(u_**2*dx)

    _test_adjoint(J, f)


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
    f = Function(V)
    f.vector()[:] = 10

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
    f = Function(V)
    f.vector()[:] = 10

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
    f = Function(V)
    f.vector()[:] = 10

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
        u_1 = Function(V)
        u_1.vector()[:] = 1

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
    h = Function(V)
    h.vector()[:] = 1
    g = Function(V)
    eps_ = [0.4/2.0**i for i in range(4)]
    residuals = []
    for eps in eps_:
        # f = bc.value()
        g.vector()[:] = f.vector()[:] + eps*h.vector()[:]
        bc.set_value(g)
        Jp = J(bc)
        tape.clear_tape()
        bc.set_value(f)
        Jm = J(bc)
        Jm.block_variable.adj_value = 1.0
        tape.evaluate_adj()

        dJdbc = bc.block_variable.adj_value

        residual = abs(Jp - Jm - eps*dJdbc.inner(h.vector()))
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)
    print(r)

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
    print(r)

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

        dJdc = c.block_variable.adj_value.vector()[0]
        print(dJdc)

        residual = abs(Jp - Jm - eps*dJdc)
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)
    print(r)

    tol = 1E-1
    assert (r[-1] > 2-tol)


def _test_adjoint(J, f):
    import numpy.random
    tape = Tape()
    set_working_tape(tape)

    V = f.function_space()
    h = Function(V)
    h.vector()[:] = numpy.random.rand(V.dim())

    eps_ = [0.01/2.0**i for i in range(5)]
    residuals = []
    for eps in eps_:

        Jp = J(f + eps*h)
        tape.clear_tape()
        Jm = J(f)
        Jm.block_variable.adj_value = 1.0
        tape.evaluate_adj()

        dJdf = f.block_variable.adj_value.vector()

        residual = abs(Jp - Jm - eps*dJdf.inner(h.vector()))
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)
    print(r)
    print(residuals)

    tol = 1E-1
    assert (r[-1] > 2-tol)
