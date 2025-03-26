import pytest

from firedrake import *
from firedrake.adjoint import *

from numpy.random import rand


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


# Tolerance in the tests.
tol = 1E-10


@pytest.mark.skipcomplex
def test_tlm_assemble():
    tape = Tape()
    set_working_tape(tape)
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    f = Function(V)
    f.vector()[:] = 5

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    u_ = Function(V)

    bc = DirichletBC(V, 1, "on_boundary")

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(h.dof_dset.size)
    g = f.copy(deepcopy=True)
    f.block_variable.tlm_value = h
    tape.evaluate_tlm()
    assert (taylor_test(Jhat, g, h, dJdm=J.block_variable.tlm_value) > 1.9)


@pytest.mark.skipcomplex
def test_tlm_bc():
    tape = Tape()
    set_working_tape(tape)
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    R = FunctionSpace(mesh, "R", 0)
    c = Function(R, val=1)
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, c, "on_boundary")

    F = inner(grad(u), grad(v)) * dx - f ** 2 * v * dx
    solve(F == 0, u, bc)

    J = assemble(c ** 2 * u * dx)
    Jhat = ReducedFunctional(J, Control(c))

    # Need to specify the domain for the constant as `ufl.action`, which requires `ufl.Constant`
    # to have a function space, will be applied on the tlm value.
    c.block_variable.tlm_value = Function(R, val=1)
    tape.evaluate_tlm()

    assert (taylor_test(Jhat, c, Function(R, val=1), dJdm=J.block_variable.tlm_value) > 1.9)


@pytest.mark.skipcomplex
def test_tlm_func():
    tape = Tape()
    set_working_tape(tape)
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    c = Function(V)
    c.vector()[:] = 1
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, c, "on_boundary")

    F = inner(grad(u), grad(v)) * dx - f ** 2 * v * dx
    solve(F == 0, u, bc)

    J = assemble(c ** 2 * u * dx)
    Jhat = ReducedFunctional(J, Control(c))

    h = Function(V)
    h.vector()[:] = rand(h.dof_dset.size)
    g = c.copy(deepcopy=True)
    c.block_variable.tlm_value = h
    tape.evaluate_tlm()

    assert (taylor_test(Jhat, g, h, dJdm=J.block_variable.tlm_value) > 1.9)


@pytest.mark.parametrize("solve_type",
                         ["solve", "LVS"])
@pytest.mark.skipcomplex
def test_time_dependent(solve_type):
    tape = Tape()
    set_working_tape(tape)
    # Defining the domain, 100 points from 0 to 1
    mesh = IntervalMesh(100, 0, 1)

    # Defining function space, test and trial functions
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    # Dirichlet boundary conditions
    bc_left = DirichletBC(V, 1, 1)
    bc_right = DirichletBC(V, 2, 2)
    bc = [bc_left, bc_right]

    # Some variables
    T = 0.5
    dt = 0.1
    f = Function(V)
    f.vector()[:] = 1

    u_1 = Function(V)
    u_1.vector()[:] = 1
    control = Control(u_1)

    a = u_1 * u * v * dx + dt * f * inner(grad(u), grad(v)) * dx
    L = u_1 * v * dx

    if solve_type == "LVS":
        problem = LinearVariationalProblem(a, L, u_, bcs=bc, constant_jacobian=False)
        solver = LinearVariationalSolver(problem)
    # Time loop
    t = dt
    while t <= T:
        if solve_type == "LVS":
            solver.solve()
        else:
            solve(a == L, u_, bc)
        u_1.assign(u_)
        t += dt

    J = assemble(u_1 ** 2 * dx)

    Jhat = ReducedFunctional(J, control)
    h = Function(V)
    h.vector()[:] = rand(h.dof_dset.size)
    u_1.tlm_value = h
    tape.evaluate_tlm()
    assert (taylor_test(Jhat, control.tape_value(), h, dJdm=J.block_variable.tlm_value) > 1.9)


@pytest.mark.skipcomplex
def test_burgers():
    tape = Tape()
    set_working_tape(tape)
    n = 30
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 2)

    def Dt(u, u_, timestep):
        return (u - u_)/timestep

    x, = SpatialCoordinate(mesh)
    pr = project(sin(2*pi*x), V)
    ic = Function(V)
    ic.vector()[:] = pr.vector()[:]

    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, ic, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    solve(F == 0, u, bc)
    u_.assign(u)
    t += float(timestep)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    end = 0.2
    while (t <= end):
        solve(F == 0, u, bc)
        u_.assign(u)

        t += float(timestep)

    J = assemble(u_*u_*dx + ic*ic*dx)

    Jhat = ReducedFunctional(J, Control(ic))
    h = Function(V)
    h.vector()[:] = rand(h.dof_dset.size)
    g = ic.copy(deepcopy=True)
    ic.block_variable.tlm_value = h
    tape.evaluate_tlm()
    assert (taylor_test(Jhat, g, h, dJdm=J.block_variable.tlm_value) > 1.9)


@pytest.mark.skipcomplex
def test_projection():
    tape = Tape()
    set_working_tape(tape)
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    bc = DirichletBC(V, Constant(1), "on_boundary")
    k = Function(R, val=2)
    x, y = SpatialCoordinate(mesh)
    expr = sin(k*x)
    f = project(expr, V)

    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)
    Jhat = ReducedFunctional(J, Control(k))

    k.block_variable.tlm_value = Constant(1)
    tape.evaluate_tlm()
    assert (taylor_test(Jhat, k, Function(R, val=1), dJdm=J.block_variable.tlm_value) > 1.9)


@pytest.mark.skipcomplex
def test_projection_function():
    tape = Tape()
    set_working_tape(tape)
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    bc = DirichletBC(V, Constant(1), "on_boundary")
    # g = Function(V)
    x, y = SpatialCoordinate(mesh)
    g = project(sin(x)*sin(y), V, annotate=False)
    expr = sin(g*x)
    f = project(expr, V)

    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)
    Jhat = ReducedFunctional(J, Control(g))

    h = Function(V)
    h.vector()[:] = rand(h.dof_dset.size)
    m = g.copy(deepcopy=True)
    g.block_variable.tlm_value = h
    tape.evaluate_tlm()
    assert (taylor_test(Jhat, m, h, dJdm=J.block_variable.tlm_value) > 1.9)
