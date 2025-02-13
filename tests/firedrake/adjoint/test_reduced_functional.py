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


@pytest.mark.skipcomplex
def test_constant():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    R = FunctionSpace(mesh, "R", 0)

    c = Function(R, val=1)
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1), "on_boundary")

    F = inner(grad(u), grad(v))*dx - f**2*v*dx
    solve(F == 0, u, bc)

    J = assemble(c**2*u*dx)
    Jhat = ReducedFunctional(J, Control(c))
    assert taylor_test(Jhat, c, Function(R, val=1)) > 1.9


@pytest.mark.skipcomplex
def test_function():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    c = Constant(1)
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v))*dx - f**2*v*dx
    solve(F == 0, u, bc)

    J = assemble(c**2*u*dx)
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dof_dset.size)
    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.skipcomplex
@pytest.mark.parametrize("control", ["dirichlet", "neumann"])
def test_wrt_function_dirichlet_boundary(control):
    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc_func = project(sin(y), V)
    bc1 = DirichletBC(V, bc_func, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1, bc2]

    g1 = Function(R, val=2)
    g2 = Function(R, val=1)
    f = Function(V)
    f.vector()[:] = 10

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx + inner(g1, v)*ds(4) + inner(g2, v)*ds(3)

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)

    if control == "dirichlet":
        Jhat = ReducedFunctional(J, Control(bc_func))
        g = bc_func
        h = Function(V)
        h.vector()[:] = 1
    else:
        Jhat = ReducedFunctional(J, Control(g1))
        g = g1
        h = Constant(1)

    assert taylor_test(Jhat, g, h) > 1.9


@pytest.mark.skipcomplex
def test_time_dependent():
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

    a = u_1*u*v*dx + dt*f*inner(grad(u), grad(v))*dx
    L = u_1*v*dx

    # Time loop
    t = dt
    while t <= T:
        solve(a == L, u_, bc)
        u_1.assign(u_)
        t += dt

    J = assemble(u_1**2*dx)

    Jhat = ReducedFunctional(J, control)

    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(Jhat, control.tape_value(), h) > 1.9


@pytest.mark.skipcomplex
def test_mixed_boundary():
    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc1 = DirichletBC(V, y**2, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1, bc2]
    g1 = Constant(2)
    g2 = Constant(1)
    f = Function(V)
    f.vector()[:] = 10

    a = f*inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx + inner(g1, v)*ds(4) + inner(g2, v)*ds(3)

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)

    Jhat = ReducedFunctional(J, Control(f))
    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.skipcomplex
def test_assemble_recompute():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    f = Function(V)
    f.vector()[:] = 2
    expr = Function(R).assign(assemble(f**2*dx))
    J = assemble(expr**2*dx(domain=mesh))
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(Jhat, f, h) > 1.9
