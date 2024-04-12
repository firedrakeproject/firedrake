import pytest
import numpy as np
from numpy.random import rand
from pyadjoint.tape import get_working_tape, pause_annotation, stop_annotating

from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import *


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    from firedrake.adjoint import annotate_tape, continue_annotation
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


@pytest.fixture(params=['iadd', 'isub'])
def op(request):
    return request.param


@pytest.fixture(params=[1, 2])
def order(request):
    return request.param


@pytest.fixture(params=[2, -1])
def power(request):
    return request.param


@pytest.fixture(params=[True, False])
def vector(request):
    return request.param


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_constant():
    mesh = UnitSquareMesh(10, 10)
    V1 = FunctionSpace(mesh, "CG", 1)

    c = Constant(1.0, domain=mesh)
    u = assemble(interpolate(c, V1))

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, Control(c))

    h = Constant(0.1, domain=mesh)
    assert taylor_test(rf, c, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_with_arguments():
    mesh = UnitSquareMesh(10, 10)
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)

    x, y = SpatialCoordinate(mesh)
    expr = x + y
    f = assemble(interpolate(expr, V1))
    interpolator = Interpolator(TestFunction(V1), V2)
    u = assemble(interpolator.interpolate(f))

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V1)
    h.vector()[:] = rand(V1.dim())
    assert taylor_test(rf, f, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_scalar_valued():
    mesh = IntervalMesh(10, 0, 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)
    V3 = FunctionSpace(mesh, "CG", 3)

    x, = SpatialCoordinate(mesh)
    f = assemble(interpolate(x, V1))
    g = assemble(interpolate(sin(x), V2))
    u = Function(V3)
    u.interpolate(3*f**2 + Constant(4.0, domain=mesh)*g)

    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V1)
    h.vector()[:] = rand(V1.dim())
    assert taylor_test(rf, f, h) > 1.9

    rf = ReducedFunctional(J, Control(g))
    h = Function(V2)
    h.vector()[:] = rand(V2.dim())
    assert taylor_test(rf, g, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_vector_valued():
    mesh = UnitSquareMesh(10, 10)
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    V2 = VectorFunctionSpace(mesh, "DG", 0)
    V3 = VectorFunctionSpace(mesh, "CG", 2)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V1))
    g = assemble(interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V2))
    u = Function(V3)
    u.interpolate(f*dot(f, g) - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V1)
    h.vector()[:] = 1
    assert taylor_test(rf, f, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_tlm():
    mesh = UnitSquareMesh(10, 10)
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    V2 = VectorFunctionSpace(mesh, "DG", 0)
    V3 = VectorFunctionSpace(mesh, "CG", 2)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V1))
    g = assemble(interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V2))
    u = Function(V3)

    u.interpolate(f - 0.5*g + f/(1+dot(f, g)))
    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V1)
    h.vector()[:] = 1
    f.block_variable.tlm_value = h

    tape = get_working_tape()
    tape.evaluate_tlm()

    assert J.block_variable.tlm_value is not None
    assert taylor_test(rf, f, h, dJdm=J.block_variable.tlm_value) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_tlm_with_constant():
    mesh = IntervalMesh(10, 0, 1)
    V1 = FunctionSpace(mesh, "CG", 2)
    V2 = FunctionSpace(mesh, "DG", 1)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(x[0], V1))
    g = assemble(interpolate(sin(x[0]), V1))
    c = Constant(5.0, domain=mesh)
    u = Function(V2)
    u.interpolate(c * f ** 2)

    # test tlm w.r.t constant only:
    c.block_variable.tlm_value = Constant(1.0, domain=mesh)
    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))
    h = Constant(1.0, domain=mesh)

    tape = get_working_tape()
    tape.evaluate_tlm()
    assert abs(J.block_variable.tlm_value - 2.0) < 1e-5
    assert taylor_test(rf, c, h, dJdm=J.block_variable.tlm_value) > 1.9

    # test tlm w.r.t constant c and function f:
    tape.reset_tlm_values()
    c.block_variable.tlm_value = Constant(0.4, domain=mesh)
    f.block_variable.tlm_value = g
    rf(c)  # replay to reset checkpoint values based on c=5
    tape.evaluate_tlm()
    assert abs(J.block_variable.tlm_value - (0.8 + 100. * (5*cos(1.) - 3*sin(1.)))) < 1e-4


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_bump_function():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)

    x, y = SpatialCoordinate(mesh)
    cx = Constant(0.5, domain=mesh)
    cy = Constant(0.5, domain=mesh)
    f = assemble(interpolate(exp(-1/(1-(x-cx)**2)-1/(1-(y-cy)**2)), V))

    J = assemble(f*y**3*dx)
    rf = ReducedFunctional(J, [Control(cx), Control(cy)])

    h = [Constant(0.1, domain=mesh), Constant(0.1, domain=mesh)]
    assert taylor_test(rf, [cx, cy], h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_self_interpolate():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

    c = Constant(1., domain=mesh)
    u.interpolate(u+c)

    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))

    h = Constant(0.1, domain=mesh)
    assert taylor_test(rf, c, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_self_interpolate_function():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

    c = Constant(1., domain=mesh)
    assemble(interpolate(u+c, V), tensor=u)
    assemble(interpolate(u+c*u**2, V), tensor=u)

    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))

    h = Constant(0.1, domain=mesh)
    assert taylor_test(rf, Constant(3., domain=mesh), h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_to_function_space():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 1)
    u = Function(V)

    x = SpatialCoordinate(mesh)
    u.interpolate(x[0])
    c = Constant(1., domain=mesh)
    w = assemble(interpolate((u+c)*u, W))

    J = assemble(w**2*dx)
    rf = ReducedFunctional(J, Control(c))
    h = Constant(0.1, domain=mesh)
    assert taylor_test(rf, Constant(1., domain=mesh), h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_to_function_space_cross_mesh():
    mesh_src = UnitSquareMesh(2, 2)
    mesh_dest = UnitSquareMesh(3, 3, quadrilateral=True)
    V = FunctionSpace(mesh_src, "CG", 1)
    W = FunctionSpace(mesh_dest, "DG", 1)
    u = Function(V)

    x = SpatialCoordinate(mesh_src)
    u.interpolate(x[0])
    c = Constant(1., domain=mesh_src)
    w = Function(W).interpolate((u+c)*u)

    J = assemble(w**2*dx)
    rf = ReducedFunctional(J, Control(c))
    h = Constant(0.1, domain=mesh_src)
    assert taylor_test(rf, Constant(1., domain=mesh_src), h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_hessian_linear_expr():
    # Note this is a direct copy of
    # pyadjoint/tests/firedrake_adjoint/test_hessian.py::test_nonlinear
    # with modifications where indicated.

    # Get tape instead of creating a new one for consistency with other tests
    tape = get_working_tape()

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Interpolate from f in another function space to force hessian evaluation
    # of interpolation. Functions in W form our control space c, our expansion
    # space h and perterbation direction g.
    W = FunctionSpace(mesh, "Lagrange", 2)
    f = Function(W)
    f.vector()[:] = 5
    # Note that we interpolate from a linear expression
    expr_interped = Function(V).interpolate(2*f)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1, domain=mesh), "on_boundary")

    F = inner(grad(u), grad(v)) * dx - u**2*v*dx - expr_interped * v * dx
    solve(F == 0, u, bc)

    J = assemble(u ** 4 * dx)
    Jhat = ReducedFunctional(J, Control(f))

    # Note functions are in W, not V.
    h = Function(W)
    h.vector()[:] = 10*rand(W.dim())

    J.block_variable.adj_value = 1.0
    f.block_variable.tlm_value = h

    tape.evaluate_adj()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0

    tape.evaluate_hessian()

    g = f.copy(deepcopy=True)

    dJdm = J.block_variable.tlm_value
    assert isinstance(f.block_variable.adj_value, Cofunction)
    assert isinstance(f.block_variable.hessian_value, Cofunction)
    Hm = f.block_variable.hessian_value.dat.inner(h.dat)
    # If the new interpolate block has the right hessian, taylor test
    # convergence rate should be as for the unmodified test.
    assert taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_hessian_nonlinear_expr():
    # Note this is a direct copy of
    # pyadjoint/tests/firedrake_adjoint/test_hessian.py::test_nonlinear
    # with modifications where indicated.

    # Get tape instead of creating a new one for consistency with other tests
    tape = get_working_tape()

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Interpolate from f in another function space to force hessian evaluation
    # of interpolation. Functions in W form our control space c, our expansion
    # space h and perterbation direction g.
    W = FunctionSpace(mesh, "Lagrange", 2)
    f = Function(W)
    f.vector()[:] = 5
    # Note that we interpolate from a nonlinear expression
    expr_interped = Function(V).interpolate(f**2)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1, domain=mesh), "on_boundary")

    F = inner(grad(u), grad(v)) * dx - u**2*v*dx - expr_interped * v * dx
    solve(F == 0, u, bc)

    J = assemble(u ** 4 * dx)
    Jhat = ReducedFunctional(J, Control(f))

    # Note functions are in W, not V.
    h = Function(W)
    h.vector()[:] = 10*rand(W.dim())

    J.block_variable.adj_value = 1.0
    f.block_variable.tlm_value = h

    tape.evaluate_adj()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0

    tape.evaluate_hessian()

    g = f.copy(deepcopy=True)

    dJdm = J.block_variable.tlm_value
    assert isinstance(f.block_variable.adj_value, Cofunction)
    assert isinstance(f.block_variable.hessian_value, Cofunction)
    Hm = f.block_variable.hessian_value.dat.inner(h.dat)
    # If the new interpolate block has the right hessian, taylor test
    # convergence rate should be as for the unmodified test.
    assert taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_hessian_nonlinear_expr_multi():
    # Note this is a direct copy of
    # pyadjoint/tests/firedrake_adjoint/test_hessian.py::test_nonlinear
    # with modifications where indicated.

    # Get tape instead of creating a new one for consistency with other tests
    tape = get_working_tape()

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Interpolate from f in another function space to force hessian evaluation
    # of interpolation. Functions in W form our control space c, our expansion
    # space h and perterbation direction g.
    W = FunctionSpace(mesh, "Lagrange", 2)
    f = Function(W)
    f.vector()[:] = 5
    w = Function(W)
    w.vector()[:] = 4
    c = Constant(2., domain=mesh)
    # Note that we interpolate from a nonlinear expression with 3 coefficients
    expr_interped = Function(V).interpolate(f**2+w**2+c**2)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1, domain=mesh), "on_boundary")

    F = inner(grad(u), grad(v)) * dx - u**2*v*dx - expr_interped * v * dx
    solve(F == 0, u, bc)

    J = assemble(u ** 4 * dx)
    Jhat = ReducedFunctional(J, Control(f))

    # Note functions are in W, not V.
    h = Function(W)
    h.vector()[:] = 10*rand(W.dim())

    J.block_variable.adj_value = 1.0
    # Note only the tlm_value of f is set here - unclear why.
    f.block_variable.tlm_value = h

    tape.evaluate_adj()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0

    tape.evaluate_hessian()

    g = f.copy(deepcopy=True)

    dJdm = J.block_variable.tlm_value
    assert isinstance(f.block_variable.adj_value, Cofunction)
    assert isinstance(f.block_variable.hessian_value, Cofunction)
    Hm = f.block_variable.hessian_value.dat.inner(h.dat)
    # If the new interpolate block has the right hessian, taylor test
    # convergence rate should be as for the unmodified test.
    assert taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_hessian_nonlinear_expr_multi_cross_mesh():
    # Note this is a direct copy of
    # pyadjoint/tests/firedrake_adjoint/test_hessian.py::test_nonlinear
    # with modifications where indicated.

    # Get tape instead of creating a new one for consistency with other tests
    tape = get_working_tape()

    mesh_dest = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh_dest, "Lagrange", 1)

    # Interpolate from f in another function space on another mesh to force
    # hessian evaluation of interpolation. Functions in W form our control
    # space c, our expansion space h and perterbation direction g.
    mesh_src = UnitSquareMesh(11, 11)
    W = FunctionSpace(mesh_src, "Lagrange", 2)
    f = Function(W)
    f.vector()[:] = 5
    w = Function(W)
    w.vector()[:] = 4
    c = Constant(2., domain=mesh_src)
    # Note that we interpolate from a nonlinear expression with 3 coefficients
    expr_interped = Function(V).interpolate(f**2+w**2+c**2)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1, domain=mesh_dest), "on_boundary")

    F = inner(grad(u), grad(v)) * dx - u**2*v*dx - expr_interped * v * dx
    solve(F == 0, u, bc)

    J = assemble(u ** 4 * dx)
    Jhat = ReducedFunctional(J, Control(f))

    # Note functions are in W, not V.
    h = Function(W)
    h.vector()[:] = 10*rand(W.dim())

    J.block_variable.adj_value = 1.0
    f.block_variable.tlm_value = h

    tape.evaluate_adj()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0

    tape.evaluate_hessian()

    g = f.copy(deepcopy=True)

    dJdm = J.block_variable.tlm_value
    assert isinstance(f.block_variable.adj_value, Cofunction)
    assert isinstance(f.block_variable.hessian_value, Cofunction)
    Hm = f.block_variable.hessian_value.dat.inner(h.dat)
    # If the new interpolate block has the right hessian, taylor test
    # convergence rate should be as for the unmodified test.
    assert taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_ioperator_replay(op, order, power):
    """
    Given source and target functions of some `order`,
    verify that replaying the tape associated with the
    augmented operators +=, -=, *= and /= gives the same
    result as a hand derivation.
    """
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", order)

    # Source and target functions
    s = assemble(interpolate(x + 1, V))
    t = assemble(interpolate(y + 1, V))
    s_orig = s.copy(deepcopy=True)
    t_orig = t.copy(deepcopy=True)
    control_s = Control(s)
    control_t = Control(t)

    # Apply the operator
    if op == 'iadd':
        t += s
    elif op == 'isub':
        t -= s
    else:
        raise ValueError("Operator '{:s}' not recognised".format(op))

    # Construct some nontrivial reduced functional
    f = lambda X: X**power
    J = assemble(f(t)*dx)
    rf_s = ReducedFunctional(J, control_s)
    rf_t = ReducedFunctional(J, control_t)

    with stop_annotating():

        # Check for consistency with the same input
        assert np.isclose(rf_s(s_orig), rf_s(s_orig))
        assert np.isclose(rf_t(t_orig), rf_t(t_orig))

        # Check for consistency with different input
        ss = s_orig.copy(deepcopy=True)
        tt = t_orig.copy(deepcopy=True)
        if op == 'iadd':
            ss += ss
            tt += tt
        elif op == 'isub':
            ss -= ss
            tt -= tt
        assert np.isclose(rf_s(t_orig), assemble(f(tt)*dx))
        assert np.isclose(rf_t(s_orig), assemble(f(ss)*dx))


def supermesh_setup(vector=False):
    fs = VectorFunctionSpace if vector else FunctionSpace
    source_mesh = UnitSquareMesh(20, 25, diagonal="left")
    source_space = fs(source_mesh, "CG", 1)
    expr = [sin(pi*xi) for xi in SpatialCoordinate(source_mesh)]
    source = assemble(interpolate(as_vector(expr) if vector else expr[0]*expr[1], source_space))
    target_mesh = UnitSquareMesh(20, 20, diagonal="right")
    target_space = fs(target_mesh, "CG", 1)
    return source, target_space


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_self_supermesh_project():
    source, target_space = supermesh_setup()
    control = Control(source)
    target = Function(target_space)
    target.project(source)
    J = assemble(target*dx)
    rf = ReducedFunctional(J, control)

    # Check forward conservation
    mass = assemble(source*dx)
    assert np.isclose(mass, J)

    # Test replay with the same input
    assert np.isclose(rf(source), J)

    # Test replay with different input
    h = Function(source)
    h.assign(10.0)
    assert np.isclose(rf(h), 10.0)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_supermesh_project_function():
    source, target_space = supermesh_setup()
    control = Control(source)
    target = Function(target_space)
    project(source, target)
    J = assemble(target*dx)
    rf = ReducedFunctional(J, control)

    # Check forward conservation
    mass = assemble(source*dx)
    assert np.isclose(mass, J)

    # Test replay with the same input
    assert np.isclose(rf(source), J)

    # Test replay with different input
    h = Function(source)
    h.assign(10.0)
    assert np.isclose(rf(h), 10.0)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_supermesh_project_to_function_space():
    source, target_space = supermesh_setup()
    control = Control(source)
    target = project(source, target_space)
    J = assemble(target*dx)
    rf = ReducedFunctional(J, control)

    # Check forward conservation
    mass = assemble(source*dx)
    assert np.isclose(mass, J)

    # Test replay with the same input
    assert np.isclose(rf(source), J)

    # Test replay with different input
    h = Function(source)
    h.assign(10.0)
    assert np.isclose(rf(h), 10.0)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_supermesh_project_gradient(vector):
    source, target_space = supermesh_setup()
    source_space = source.function_space()
    control = Control(source)
    target = project(source, target_space)
    J = assemble(inner(target, target)*dx)
    rf = ReducedFunctional(J, control)

    # Taylor test
    h = Function(source_space)
    h.vector()[:] = rand(source_space.dim())
    minconv = taylor_test(rf, source, h)
    assert minconv > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_supermesh_project_tlm(vector):
    source, target_space = supermesh_setup()
    control = Control(source)
    target = project(source, target_space)
    J = assemble(inner(target, target)*dx)
    rf = ReducedFunctional(J, control)

    # Test replay with different input
    h = Function(source)
    h.assign(1.0)
    source.block_variable.tlm_value = h

    tape = get_working_tape()
    tape.evaluate_tlm()

    assert J.block_variable.tlm_value is not None
    assert taylor_test(rf, source, h, dJdm=J.block_variable.tlm_value) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_supermesh_project_hessian(vector):
    source, target_space = supermesh_setup()
    control = Control(source)
    target = project(source, target_space)
    J = assemble(inner(target, target)**2*dx)
    rf = ReducedFunctional(J, control)

    source_space = source.function_space()
    h = Function(source_space)
    h.vector()[:] = 10*rand(source_space.dim())

    J.block_variable.adj_value = 1.0
    source.block_variable.tlm_value = h

    tape = get_working_tape()
    tape.evaluate_adj()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0.0

    tape.evaluate_hessian()

    dJdm = J.block_variable.tlm_value
    assert isinstance(source.block_variable.adj_value, Cofunction)
    assert isinstance(source.block_variable.hessian_value, Cofunction)
    Hm = source.block_variable.hessian_value.dat.inner(h.dat)
    assert taylor_test(rf, source, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_init_constant():
    mesh = UnitSquareMesh(1, 1)
    c1 = Constant(1.0, domain=mesh)
    c2 = Constant(0.0, domain=mesh)
    c2.assign(c1)
    J = assemble(c2*dx(domain=mesh))
    rf = ReducedFunctional(J, Control(c1))
    assert np.isclose(rf(-1.0), -1.0)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_init_constant_diff_mesh():
    mesh = UnitSquareMesh(1, 1)
    mesh0 = UnitSquareMesh(2, 2)
    c1 = Constant(1.0, domain=mesh)
    c2 = Constant(0.0, domain=mesh0)
    c2.assign(c1)
    J = assemble(c2*dx(domain=mesh0))
    rf = ReducedFunctional(J, Control(c1))
    assert np.isclose(rf(Constant(-1.0, domain=mesh)), -1.0)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_copy_function():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    one = Constant(1.0, domain=mesh)
    f = assemble(interpolate(one, V))
    g = f.copy(deepcopy=True)
    J = assemble(g*dx)
    rf = ReducedFunctional(J, Control(f))
    a = assemble(Interpolate(-one, V))
    assert np.isclose(rf(a), -J)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_consecutive_nonlinear_solves():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    uic = Constant(2.0, domain=mesh)
    u1 = Function(V).assign(uic)
    u0 = Function(u1)
    v = TestFunction(V)
    F = v * u1**2 * dx - v*u0 * dx
    problem = NonlinearVariationalProblem(F, u1)
    solver = NonlinearVariationalSolver(problem)
    for i in range(3):
        u0.assign(u1)
        solver.solve()
    J = assemble(u1**16*dx)
    rf = ReducedFunctional(J, Control(uic))
    h = Constant(0.01, domain=mesh)
    assert taylor_test(rf, uic, h) > 1.9


@pytest.mark.skipcomplex
def test_assign_function():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    uic = Function(V, name="uic").assign(1.0)
    u0 = Function(V, name="u0")
    u1 = Function(V, name="u1")
    u0.assign(uic)
    u1.assign(2 * u0 + uic)
    J = assemble(((u1 + Constant(1.0)) ** 2) * dx)
    rf = ReducedFunctional(J, Control(uic))
    h = Function(V, name="h").assign(0.01)
    assert taylor_test(rf, uic, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_3325():
    # See https://github.com/firedrakeproject/firedrake/issues/3325
    # for the original MFE, this has been simplified
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(x * y)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * dx
    L = inner(f, v) * dx
    bc = DirichletBC(V, Constant(1), "on_boundary")
    sol = Function(V)

    solve(a == L, sol, bcs=bc)

    g = Function(V, name="control")
    J = assemble(1./2*inner(grad(sol), grad(sol))*dx + inner(g, g)*ds(4))
    control = Control(g)
    Jhat = ReducedFunctional(J, control)

    constraint = UFLInequalityConstraint(-inner(g, g)*ds(4), control)
    minimize(Jhat, method="SLSQP", constraints=constraint)
