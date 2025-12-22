import pytest
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from pytest_mpi.parallel_assert import parallel_assert


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
    f.assign(1.)

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
    f.assign(1.)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v))*dx - f**2*v*dx
    solve(F == 0, u, bc)

    J = assemble(c**2*u*dx)
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.dat.data[:] = np.random.rand(V.dof_dset.size)
    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.skipcomplex
@pytest.mark.parametrize("control", ["dirichlet-pre", "dirichlet-post", "neumann"])
def test_wrt_function_dirichlet_boundary(control):
    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc_func = project(sin(y), V)
    bc1 = DirichletBC(V, bc_func, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1, bc2]

    g1 = Function(R, val=2)
    g2 = Function(R, val=1)
    f = Function(V)
    f.assign(10.)

    a = inner(grad(u), grad(v))*dx + u**2*v*dx
    L = inner(f, v)*dx + inner(g1, v)*ds(4) + inner(g2, v)*ds(3)

    pre_apply_bcs = (control == "dirichlet-post")
    solve(a - L == 0, u, bc, pre_apply_bcs=pre_apply_bcs)

    J = assemble(u**2*dx)

    if control.startswith("dirichlet"):
        Jhat = ReducedFunctional(J, Control(bc_func))
        g = bc_func
        h = Function(V)
        h.assign(1.)
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
    f.assign(1.)

    u_1 = Function(V)
    u_1.assign(1.)
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
    h.assign(1.)
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
    f.assign(10.)

    a = f*inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx + inner(g1, v)*ds(4) + inner(g2, v)*ds(3)

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)

    Jhat = ReducedFunctional(J, Control(f))
    h = Function(V)
    h.assign(1.)
    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.skipcomplex
def test_assemble_recompute():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    f = Function(V)
    f.assign(2.)
    expr = Function(R).assign(assemble(f**2*dx))
    J = assemble(expr**2*dx(domain=mesh))
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.assign(1.)
    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.skipcomplex
def test_interpolate():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    c = Cofunction(Q.dual())
    c.dat.data[:] = 1

    f = Function(V)
    f.dat.data[:] = 2
    J = assemble(interpolate(f**2, c))
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.dat.data[:] = 3
    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.skipcomplex
def test_interpolate_mixed():
    mesh = UnitSquareMesh(2, 2)
    V1 = FunctionSpace(mesh, "RT", 1)
    V2 = FunctionSpace(mesh, "CG", 1)
    V = V1 * V2

    Q1 = FunctionSpace(mesh, "DG", 0)
    Q2 = FunctionSpace(mesh, "N1curl", 1)
    Q = Q1 * Q2

    c = Cofunction(Q.dual())
    c.subfunctions[0].dat.data[:] = 1
    c.subfunctions[1].dat.data[:] = 2

    f = Function(V)
    f.subfunctions[0].dat.data[:] = 3
    f.subfunctions[1].dat.data[:] = 4

    f1, f2 = split(f)
    exprs = [f2 * div(f1)**2, grad(f2) * div(f1)]
    expr = as_vector([e[i] for e in exprs for i in np.ndindex(e.ufl_shape)])
    J = assemble(interpolate(expr, c))
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.subfunctions[0].dat.data[:] = 5
    h.subfunctions[1].dat.data[:] = 6
    assert taylor_test(Jhat, f, h) > 1.9


@pytest.mark.skipcomplex
@pytest.mark.parallel(2)
def test_real_space_assign_numpy():
    """Check that Function._ad_assign_numpy correctly handles
    zero length arrays on some ranks for Real space in parallel.
    """
    mesh = UnitSquareMesh(1, 1)
    R = FunctionSpace(mesh, "R", 0)
    dst = Function(R)
    src = dst.dat.dataset.layout_vec.array_r.copy()
    data = 1 + np.arange(src.shape[0])
    src[:] = data
    dst._ad_assign_numpy(dst, src, offset=0)
    parallel_assert(np.allclose(dst.dat.data_ro, data))


@pytest.mark.skipcomplex
@pytest.mark.parallel(2)
def test_real_space_parallel():
    """Check that scipy.optimize works for Real space in parallel
    despite dat.data array having zero length on some ranks.
    """
    mesh = UnitSquareMesh(1, 1)
    R = FunctionSpace(mesh, "R", 0)
    m = Function(R)
    J = assemble((m-1)**2*dx)
    Jhat = ReducedFunctional(J, Control(m))
    opt = minimize(Jhat)
    parallel_assert(np.allclose(opt.dat.data_ro, 1))
