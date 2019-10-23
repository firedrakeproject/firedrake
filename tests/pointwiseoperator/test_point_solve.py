import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


def test_properties(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    w = Function(V)
    g = Function(V)

    def _check_extop_attributes_(x, ops, space, der, shape):
        assert x.ufl_function_space() == space
        assert x.ufl_operands == ops
        assert x.derivatives == der
        assert x.ufl_shape == shape

    f = lambda x, y, z: x**2*y - z
    fprime = lambda x, y, z: 2*x*y
    fprime2 = lambda x, y, z: 2*y
    solver_params = {'fprime': fprime, 'args': (), 'tol': 1.0e-07, 'maxiter': 10, 'fprime2': fprime2,
                     'x1': g, 'rtol': 1.0, 'full_output': True, 'disp': False}
    ps = point_solve(f, function_space=V, solver_name='newton', disp=True, solver_params=solver_params)
    ps2 = ps(w, g)

    _check_extop_attributes_(ps2, (w, g), V, (0, 0), ())

    operator_data = {'point_solve': f, 'solver_name': 'newton', 'solver_params': solver_params}
    assert ps2.operator_data == operator_data
    assert ps2.solver_params == solver_params
    assert ps2.operator_f == f
    assert ps2.disp


def test_pointwise_solve_operator(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    uexact = Function(P).assign(1)

    m = uexact
    a1 = m*dx

    a = Function(V).assign(0)
    b = Function(V).assign(1)

    # Conflict with ufl if we use directly cos()
    p = point_solve(lambda x, y, m1, m2: (1-m1)*(1-x)**2 + 100*m2*(y-x**2)**2, function_space=P)
    p2 = p(b, a, b)  # Rosenbrock function for (m1,m2) = (0,1), the global minimum is reached in (1,1)
    a2 = p2*dx

    assert p2.ufl_operands == (b, a, b)
    assert p2._ufl_function_space == P
    assert p2.derivatives == (0, 0, 0)
    assert p2.ufl_shape == ()

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    assert abs(assemble_a1 - assemble_a2) < 1.0e-7
    assemble(p2*dx)

    u = Function(V)
    u2 = Function(V)
    u3 = Function(V)
    v = TestFunction(V)
    g = Function(V).assign(1.)

    f = Function(V).interpolate(cos(x)*sin(y))
    p = point_solve(lambda x, y: x**3-y, function_space=V, solver_params={'x0': g+0.3})
    p2 = p(g)

    F = (dot(grad(p2*u), grad(v)) + u*v)*dx - f*v*dx
    solve(F == 0, u)

    F = (dot(grad((g**2)*u2), grad(v)) + u2*v)*dx - f*v*dx
    solve(F == 0, u2)

    F = (dot(grad(p2*u3), grad(v)) + u3*v)*dx - f*v*dx
    problem = NonlinearVariationalProblem(F, u3)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()

    a1 = assemble(u*dx)
    a2 = assemble(u2*dx)
    a3 = assemble(u3*dx)
    err = (a1-a2)**2 + (a2-a3)**2
    assert err < 1.0e-9


def test_compute_derivatives(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    a = Function(V).assign(0)
    b = Function(V).assign(1)

    x0 = Function(P).assign(1.1)
    p = point_solve(lambda x, y, m1, m2: x - y**2 + m1*m2, function_space=P, solver_params={'x0': x0})
    p2 = p(b, a, a)
    dp2db = p2._ufl_expr_reconstruct_(b, a, a, derivatives=(1, 0, 0))
    a3 = dp2db*dx

    a4 = 2*b*dx  # dp2/db

    assemble_a3 = assemble(a3)
    assemble_a4 = assemble(a4)

    assert abs(assemble_a3 - assemble_a4) < 1.0e-7


def test_scalar_check_equality(mesh):

    V1 = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    f = Function(V1).interpolate(cos(x)*sin(y))

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    ps = point_solve(lambda x, y: x - y, function_space=V1, solver_params={'maxiter': 50})
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_solve = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_solve < 1.0e-09


def test_vector_check_equality(mesh):

    V1 = VectorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    f = Function(V1).interpolate(as_vector([cos(x), sin(y)]))

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    ps = point_solve(lambda x, y: x - y, function_space=V1, solver_params={'maxiter': 50})
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_solve = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_solve < 1.0e-09


def test_tensor_check_equality(mesh):

    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = TensorFunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    phi = Function(V0).interpolate(as_vector([cos(x), sin(y)]))
    f = grad(phi)

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    ps = point_solve(lambda x, y: x - y, function_space=V1, solver_params={'maxiter': 50})
    tau2 = ps(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_solve = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_solve < 1.0e-09


def test_sym_grad_check_equality(mesh):

    x, y = SpatialCoordinate(mesh)

    V1 = VectorFunctionSpace(mesh, "CG", 1)

    f = Function(V1).interpolate(as_vector([cos(x), sin(y)]))

    w = TestFunction(V1)

    u = Function(V1)

    F = inner(grad(w), sym(grad(u)))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u3 = Function(V1).assign(1.5)

    u2 = Function(V1)
    ps = point_solve(lambda x, y: x - y, function_space=V1, solver_name='newton', solver_params={'maxiter': 50, 'x0': u2+u3})
    # ps = point_solve(lambda x, y: x - y, function_space=V1, solver_name='halley' ,solver_params={'maxiter':50, 'x1':u2}, disp=True)
    tau2 = ps(u2)

    F2 = inner(grad(w), sym(grad(u2)))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err < 1.0e-09
