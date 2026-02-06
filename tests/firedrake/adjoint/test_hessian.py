import pytest

from firedrake import *
from firedrake.adjoint import *


@pytest.fixture(autouse=True)
def test_taping(set_test_tape):
    pass


@pytest.fixture(autouse=True, scope="module")
def module_annotation(set_module_annotation):
    pass


@pytest.fixture
def rg():
    return RandomGenerator(PCG64(seed=1234))


@pytest.mark.skipcomplex
def test_simple_solve(rg):
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V).assign(2.)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx
    L = f*v*dx

    u_ = Function(V)

    solve(a == L, u_)

    L = u_*v*dx

    u_sol = Function(V)
    solve(a == L, u_sol)

    J = assemble(u_sol**4*dx)
    c = Control(f)
    Jhat = ReducedFunctional(J, c)

    h = rg.uniform(V)

    tape.evaluate_adj()

    m = f.copy(deepcopy=True)
    dJdm = assemble(inner(Jhat.derivative(apply_riesz=True), h)*dx)
    Hm = assemble(inner(Jhat.hessian(h, apply_riesz=True), h)*dx)
    assert taylor_test(Jhat, m, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipcomplex
def test_mixed_derivatives(rg):
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V).assign(2.)
    control_f = Control(f)

    g = Function(V).assign(3.)
    control_g = Control(g)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = f**2*u*v*dx
    L = g**2*v*dx

    u_ = Function(V)
    solve(a == L, u_)

    J = assemble(u_**2*dx)
    Jhat = ReducedFunctional(J, [control_f, control_g])

    # Direction to take a step for convergence test
    h = rg.uniform(V)

    # Evaluate TLM
    control_f.tlm_value = h
    control_g.tlm_value = h
    tape.evaluate_tlm()

    # Evaluate Adjoint
    J.block_variable.adj_value = 1.0
    tape.evaluate_adj()

    # Evaluate Hessian
    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    dJdm = J.block_variable.tlm_value
    Hm = control_f.hessian_value.dat.inner(h.dat) + control_g.hessian_value.dat.inner(h.dat)

    assert taylor_test(Jhat, [f, g], [h, h], dJdm, Hm) > 2.9


@pytest.mark.skipcomplex
def test_function(rg):
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 2)
    R = FunctionSpace(mesh, "R", 0)
    c = Function(R, val=4)
    control_c = Control(c)
    f = Function(V).assign(3.)
    control_f = Control(f)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1), "on_boundary")

    F = inner(grad(u), grad(v)) * dx + u**2*v*dx - f ** 2 * v * dx - c**2*v*dx
    solve(F == 0, u, bc)

    J = assemble(c ** 2 * u ** 2 * dx)

    Jhat = ReducedFunctional(J, [control_c, control_f])
    dJdc, dJdf = compute_derivative(J, [control_c, control_f], apply_riesz=True)

    # Step direction for derivatives and convergence test
    h_c = Function(R, val=1.0)
    h_f = rg.uniform(V, 0, 10)

    # Total derivative
    dJdc, dJdf = compute_derivative(J, [control_c, control_f], apply_riesz=True)
    dJdm = assemble(dJdc * h_c * dx + dJdf * h_f * dx)

    # Hessian
    Hcc, Hff = compute_hessian(J, [control_c, control_f], [h_c, h_f], apply_riesz=True)
    Hm = assemble(Hcc * h_c * dx + Hff * h_f * dx)
    assert taylor_test(Jhat, [c, f], [h_c, h_f], dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipcomplex
def test_nonlinear(rg):
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "Lagrange", 1)
    R = FunctionSpace(mesh, "R", 0)
    f = Function(V).assign(5.)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1), "on_boundary")

    F = inner(grad(u), grad(v)) * dx - u**2*v*dx - f * v * dx
    solve(F == 0, u, bc)

    J = assemble(u ** 4 * dx)
    Jhat = ReducedFunctional(J, Control(f))

    h = rg.uniform(V, 0, 10)

    J.block_variable.adj_value = 1.0
    f.block_variable.tlm_value = h

    tape.evaluate_adj()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    g = f.copy(deepcopy=True)

    dJdm = J.block_variable.tlm_value
    Hm = f.block_variable.hessian_value.dat.inner(h.dat)
    assert taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.8


@pytest.mark.skipcomplex
def test_dirichlet(rg):
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V).assign(30.)

    u = Function(V)
    v = TestFunction(V)
    c = Function(V).assign(1.)
    bc = DirichletBC(V, c, "on_boundary")

    F = inner(grad(u), grad(v)) * dx + u**4*v*dx - f**2 * v * dx
    solve(F == 0, u, bc, solver_parameters={"snes_rtol": 1e-10})

    J = assemble(u ** 4 * dx)
    Jhat = ReducedFunctional(J, Control(c))

    h = rg.uniform(V)

    J.block_variable.adj_value = 1.0
    c.block_variable.tlm_value = h

    tape.evaluate_adj()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    g = c.copy(deepcopy=True)

    dJdm = J.block_variable.tlm_value

    Hm = c.block_variable.hessian_value.dat.inner(h.dat)
    assert taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipcomplex
@pytest.mark.parametrize("solve_type", ["solve", "nlvs"])
def test_burgers(solve_type, rg):
    tape = Tape()
    set_working_tape(tape)
    n = 100
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 2)

    def Dt(u, u_, timestep):
        return (u - u_)/timestep

    x, = SpatialCoordinate(mesh)
    pr = project(sin(2*pi*x), V, annotate=False)
    ic = Function(V).assign(pr)

    u_ = Function(V).assign(ic)
    u = Function(V).assign(ic)
    v = TestFunction(V)

    nu = Constant(0.0001)

    dt = 0.01
    nt = 20

    params = {
        'snes_rtol': 1e-10,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }

    F = (Dt(u, u_, dt)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    if solve_type == "nlvs":
        use_nlvs = True
    elif solve_type == "solve":
        use_nlvs = False
    else:
        raise ValueError(f"Unrecognised solve type {solve_type}")

    if use_nlvs:
        solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(F, u),
            solver_parameters=params)

    if use_nlvs:
        solver.solve()
    else:
        solve(F == 0, u, bc, solver_parameters=params)
    u_.assign(u)

    for _ in range(nt):
        if use_nlvs:
            solver.solve()
        else:
            solve(F == 0, u, bc, solver_parameters=params)
        u_.assign(u)

    J = assemble(u_*u_*dx + ic*ic*dx)

    Jhat = ReducedFunctional(J, Control(ic))
    h = rg.uniform(V)
    g = ic.copy(deepcopy=True)
    J.block_variable.adj_value = 1.0
    ic.block_variable.tlm_value = h
    tape.evaluate_adj()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    dJdm = J.block_variable.tlm_value
    Hm = ic.block_variable.hessian_value.dat.inner(h.dat)
    assert taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9
