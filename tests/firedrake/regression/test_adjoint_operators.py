import pytest
import numpy as np
from numpy.random import rand
from pyadjoint.tape import get_working_tape, pause_annotation, stop_annotating
from ufl.classes import Zero

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
    R = FunctionSpace(mesh, "R", 0)
    c = Function(R, val=1.0)
    u = assemble(interpolate(c, V1))

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, Control(c))

    assert taylor_test(rf, c, Function(R, val=0.1)) > 1.9


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
    R = FunctionSpace(mesh, "R", 0)

    x, = SpatialCoordinate(mesh)
    f = assemble(interpolate(x, V1))
    g = assemble(interpolate(sin(x), V2))
    u = Function(V3)
    u.interpolate(3*f**2 + Function(R, val=4.0) * g)

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
    R = FunctionSpace(mesh, "R", 0)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(x[0], V1))
    g = assemble(interpolate(sin(x[0]), V1))
    c = Function(R, val=5.0)
    u = Function(V2)
    u.interpolate(c * f ** 2)

    # test tlm w.r.t constant only:
    c.block_variable.tlm_value = Function(R, val=1.0)
    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))

    tape = get_working_tape()
    tape.evaluate_tlm()
    assert abs(J.block_variable.tlm_value - 2.0) < 1e-5
    assert taylor_test(rf, c, Function(R, val=1.0), dJdm=J.block_variable.tlm_value) > 1.9

    # test tlm w.r.t constant c and function f:
    tape.reset_tlm_values()
    c.block_variable.tlm_value = Function(R, val=0.4)
    f.block_variable.tlm_value = g
    rf(c)  # replay to reset checkpoint values based on c=5
    tape.evaluate_tlm()
    assert abs(J.block_variable.tlm_value - (0.8 + 100. * (5*cos(1.) - 3*sin(1.)))) < 1e-4


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_bump_function():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)
    R = FunctionSpace(mesh, "R", 0)

    x, y = SpatialCoordinate(mesh)
    cx = Function(R, val=0.5)
    cy = Function(R, val=0.5)
    f = assemble(interpolate(exp(-1/(1-(x-cx)**2)-1/(1-(y-cy)**2)), V))

    J = assemble(f*y**3*dx)
    rf = ReducedFunctional(J, [Control(cx), Control(cy)])

    h = [Function(R, val=0.1), Function(R, val=0.1)]
    assert taylor_test(rf, [cx, cy], h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_self_interpolate():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = Function(V)

    c = Function(R, val=1.0)
    u.interpolate(u+c)

    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))
    assert taylor_test(rf, c, Function(R, val=0.1)) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_self_interpolate_function():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = Function(V)

    c = Function(R, val=1.0)
    assemble(interpolate(u+c, V), tensor=u)
    assemble(interpolate(u+c*u**2, V), tensor=u)

    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))
    assert taylor_test(rf, Function(R, val=3.0), Function(R, val=0.1)) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_to_function_space():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = Function(V)

    x = SpatialCoordinate(mesh)
    u.interpolate(x[0])
    c = Function(R, val=1.0)
    w = assemble(interpolate((u+c)*u, W))

    J = assemble(w**2*dx)
    rf = ReducedFunctional(J, Control(c))
    assert taylor_test(rf, Function(R, val=1.0), Function(R, val=0.1)) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_to_function_space_cross_mesh():
    mesh_src = UnitSquareMesh(2, 2)
    mesh_dest = UnitSquareMesh(3, 3, quadrilateral=True)
    V = FunctionSpace(mesh_src, "CG", 1)
    W = FunctionSpace(mesh_dest, "DG", 1)
    R = FunctionSpace(mesh_src, "R", 0)
    u = Function(V)

    x = SpatialCoordinate(mesh_src)
    u.interpolate(x[0])
    c = Function(R, val=1.0)
    w = Function(W).interpolate((u+c)*u)

    J = assemble(w**2*dx)
    rf = ReducedFunctional(J, Control(c))
    assert taylor_test(rf, Function(R, val=1.0), Function(R, val=0.1)) > 1.9


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
    R = FunctionSpace(mesh, "R", 0)
    f = Function(W)
    f.vector()[:] = 5
    # Note that we interpolate from a linear expression
    expr_interped = Function(V).interpolate(2*f)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1.0), "on_boundary")

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
    R = FunctionSpace(mesh, "R", 0)
    f = Function(W)
    f.vector()[:] = 5
    # Note that we interpolate from a nonlinear expression
    expr_interped = Function(V).interpolate(f**2)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1.0), "on_boundary")

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
    R = FunctionSpace(mesh, "R", 0)
    f = Function(W)
    f.vector()[:] = 5
    w = Function(W)
    w.vector()[:] = 4
    c = Function(R, val=2.0)
    # Note that we interpolate from a nonlinear expression with 3 coefficients
    expr_interped = Function(V).interpolate(f**2+w**2+c**2)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R, val=1.0), "on_boundary")

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
    R_dest = FunctionSpace(mesh_dest, "R", 0)

    # Interpolate from f in another function space on another mesh to force
    # hessian evaluation of interpolation. Functions in W form our control
    # space c, our expansion space h and perterbation direction g.
    mesh_src = UnitSquareMesh(11, 11)
    R_src = FunctionSpace(mesh_src, "R", 0)
    W = FunctionSpace(mesh_src, "Lagrange", 2)
    f = Function(W)
    f.vector()[:] = 5
    w = Function(W)
    w.vector()[:] = 4
    c = Function(R_src, val=2.0)
    # Note that we interpolate from a nonlinear expression with 3 coefficients
    expr_interped = Function(V).interpolate(f**2+w**2+c**2)

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Function(R_dest, val=1.0), "on_boundary")

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
    R = FunctionSpace(mesh, "R", 0)
    c1 = Function(R, val=1.0)
    c2 = Function(R, val=0.0)
    c2.assign(c1)
    J = assemble(c2*dx(domain=mesh))
    rf = ReducedFunctional(J, Control(c1))
    assert np.isclose(rf(Function(R, val=-1.0)), -1.0)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_init_constant_diff_mesh():
    mesh = UnitSquareMesh(1, 1)
    mesh0 = UnitSquareMesh(2, 2)
    R = FunctionSpace(mesh, "R", 0)
    R0 = FunctionSpace(mesh0, "R", 0)
    c1 = Function(R, val=1.0)
    c2 = Function(R0, val=0.0)
    c2.assign(c1)
    J = assemble(c2*dx(domain=mesh0))
    rf = ReducedFunctional(J, Control(c1))
    assert np.isclose(rf(Function(R, val=-1.0)), -1.0)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_copy_function():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    one = Function(R, val=1.0)
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
    R = FunctionSpace(mesh, "R", 0)
    uic = Function(R, val=2.0)
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
    assert taylor_test(rf, uic, Function(R, val=0.01)) > 1.9


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


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.parametrize("solve_type", ["solve", "linear_variational_solver"])
def test_assign_cofunction(solve_type):
    # See https://github.com/firedrakeproject/firedrake/issues/3464 .
    # This function tests the case where Cofunction assigns a
    # Cofunction and a BaseForm.
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)
    k = Function(V).assign(1.0)
    a = k * u * v * dx
    b = Constant(1.0) * v * dx
    u0 = Cofunction(V.dual(), name="u0")
    u1 = Cofunction(V.dual(), name="u1")
    sol = Function(V, name="sol")
    if solve_type == "linear_variational_solver":
        problem = LinearVariationalProblem(lhs(a), rhs(a) + u1, sol)
        solver = LinearVariationalSolver(problem)
    J = 0
    for i in range(2):
        # This loop emulates a time-dependent problem, where the Cofunction
        # added on the right-hand of the equation is updated at each time step.
        u0.assign(assemble(b))
        u1.assign(i * u0 + b)
        if solve_type == "solve":
            solve(a == u1, sol)
        if solve_type == "linear_variational_solver":
            solver.solve()
        J += assemble(((sol + Constant(1.0)) ** 2) * dx)
    rf = ReducedFunctional(J, Control(k))
    assert np.isclose(rf(k), J, rtol=1e-10)
    assert taylor_test(rf, k, Function(V).assign(0.1)) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_assign_zero_cofunction():
    # See https://github.com/firedrakeproject/firedrake/issues/3464 .
    # It is expected the tape breaks since the functional loses its dependency
    # on the control after the Cofunction assigns Zero.
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)
    k = Function(V).assign(1.0)
    a = u * v * dx
    b = k * v * dx
    u0 = Cofunction(V.dual(), name="u0")
    u0.assign(b)
    u0.assign(Zero())
    sol = Function(V, name="c")
    solve(a == u0, sol)
    J = assemble(((sol + Constant(1.0)) ** 2) * dx)
    # The zero assignment should break the tape and hence cause a zero
    # gradient.
    grad_l2 = compute_gradient(J, Control(k), options={"riesz_representation": "l2"})
    grad_none = compute_gradient(J, Control(k), options={"riesz_representation": None})
    grad_h1 = compute_gradient(J, Control(k), options={"riesz_representation": "H1"})
    grad_L2 = compute_gradient(J, Control(k), options={"riesz_representation": "L2"})
    assert isinstance(grad_l2, Function) and isinstance(grad_L2, Function) \
        and isinstance(grad_h1, Function)
    assert isinstance(grad_none, Cofunction)
    assert all(grad_none.dat.data_ro == 0.0)
    assert all(grad_l2.dat.data_ro == 0.0)
    assert all(grad_h1.dat.data_ro == 0.0)
    assert all(grad_L2.dat.data_ro == 0.0)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_cofunction_subfunctions_with_adjoint():
    # See https://github.com/firedrakeproject/firedrake/issues/3469
    mesh = UnitSquareMesh(2, 2)
    BDM = FunctionSpace(mesh, "BDM", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    W = BDM * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    x, y = SpatialCoordinate(mesh)
    f = Function(DG).interpolate(
        10*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))
    bc0 = DirichletBC(W.sub(0), as_vector([0.0, -sin(5*x)]), 3)
    bc1 = DirichletBC(W.sub(0), as_vector([0.0, sin(5*x)]), 4)
    k = Function(DG).assign(1.0)
    a = (dot(sigma, tau) + (dot(div(tau), u))) * dx + k * div(sigma)*v*dx
    b = assemble(-f*TestFunction(DG)*dx)
    w = Function(W)
    b1 = Cofunction(W.dual())
    # The following operation generates the FunctionMergeBlock.
    b1.sub(1).interpolate(b)
    solve(a == b1, w, bcs=[bc0, bc1])
    J = assemble(0.5*dot(w, w)*dx)
    J_hat = ReducedFunctional(J, Control(k))
    k.block_variable.tlm_value = Constant(1)
    get_working_tape().evaluate_tlm()
    assert taylor_test(J_hat, k, Constant(1.0), dJdm=J.block_variable.tlm_value) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_riesz_representation_for_adjoints():
    # Check if the Riesz representation norms for adjoints are working as expected.
    mesh = UnitIntervalMesh(1)
    space = FunctionSpace(mesh, "Lagrange", 1)
    f = Function(space).interpolate(SpatialCoordinate(mesh)[0])
    J = assemble((f ** 2) * dx)
    rf = ReducedFunctional(J, Control(f))
    with stop_annotating():
        v = TestFunction(space)
        u = TrialFunction(space)
        dJdu_cofunction = assemble(derivative((f ** 2) * dx, f, v))

        # Riesz representation with l2
        dJdu_function_l2 = Function(space, val=dJdu_cofunction.dat)

        # Riesz representation with H1
        a = u * v * dx + inner(grad(u), grad(v)) * dx
        dJdu_function_H1 = Function(space)
        solve(a == dJdu_cofunction, dJdu_function_H1)

        # Riesz representation with L2
        a = u*v*dx
        dJdu_function_L2 = Function(space)
        solve(a == dJdu_cofunction, dJdu_function_L2)

    dJdu_none = rf.derivative(options={"riesz_representation": None})
    dJdu_l2 = rf.derivative(options={"riesz_representation": "l2"})
    dJdu_H1 = rf.derivative(options={"riesz_representation": "H1"})
    dJdu_L2 = rf.derivative(options={"riesz_representation": "L2"})
    dJdu_default_L2 = rf.derivative()
    assert (
        isinstance(dJdu_none, Cofunction) and isinstance(dJdu_function_l2, Function)
        and isinstance(dJdu_H1, Function) and isinstance(dJdu_default_L2, Function)
        and isinstance(dJdu_L2, Function)
        and np.allclose(dJdu_none.dat.data, dJdu_cofunction.dat.data)
        and np.allclose(dJdu_l2.dat.data, dJdu_function_l2.dat.data)
        and np.allclose(dJdu_H1.dat.data, dJdu_function_H1.dat.data)
        and np.allclose(dJdu_default_L2.dat.data, dJdu_function_L2.dat.data)
        and np.allclose(dJdu_L2.dat.data, dJdu_function_L2.dat.data)
    )


@pytest.mark.skipcomplex
@pytest.mark.parametrize("constant_jacobian", [False, True])
def test_lvs_constant_jacobian(constant_jacobian):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)
    trial = TrialFunction(space)

    u = Function(space, name="u").interpolate(X[0] - 0.5)
    with stop_annotating():
        u_ref = u.copy(deepcopy=True)
    v = Function(space, name="v")
    problem = LinearVariationalProblem(
        inner(trial, test) * dx, inner(u, test) * dx, v,
        constant_jacobian=constant_jacobian)
    solver = LinearVariationalSolver(problem)
    solver.solve()
    J = assemble(v * v * dx)

    J_hat = ReducedFunctional(J, Control(u))

    dJ = J_hat.derivative(options={"riesz_representation": None})
    assert np.allclose(dJ.dat.data_ro, 2 * assemble(inner(u_ref, test) * dx).dat.data_ro)

    u_ref = Function(space, name="u").interpolate(X[0] - 0.1)
    J_hat(u_ref)

    dJ = J_hat.derivative(options={"riesz_representation": None})
    assert np.allclose(dJ.dat.data_ro, 2 * assemble(inner(u_ref, test) * dx).dat.data_ro)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_cofunction_assign_functional():
    """Test that cofunction assignment is correctly annotated.
    """
    mesh = UnitIntervalMesh(5)
    fs = FunctionSpace(mesh, "R", 0)
    f = Function(fs)
    f.assign(1.0)
    f2 = Function(fs)
    f2.assign(1.0)
    v = TestFunction(fs)

    cof = assemble(f * v * dx)
    cof2 = Cofunction(cof)
    cof2.assign(cof)  # Test is checking that this is taped.
    J = assemble(action(cof2, f2))
    Jhat = ReducedFunctional(J, Control(f))
    assert np.allclose(float(Jhat.derivative()), 1.0)
    f.assign(2.0)
    assert np.allclose(Jhat(f), 2.0)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_bdy_control():
    # Test for the case the boundary condition is a control for a
    # domain with length different from 1.
    mesh = IntervalMesh(10, 0, 2)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)
    trial = TrialFunction(space)
    sol = Function(space, name="sol")
    # Dirichlet boundary conditions
    R = FunctionSpace(mesh, "R", 0)
    a = Function(R, val=1.0)
    b = Function(R, val=2.0)
    bc_left = DirichletBC(space, a, 1)
    bc_right = DirichletBC(space, b, 2)
    bc = [bc_left, bc_right]
    F = dot(grad(trial), grad(test)) * dx
    problem = LinearVariationalProblem(lhs(F), rhs(F), sol, bcs=bc)
    solver = LinearVariationalSolver(problem)
    solver.solve()
    # Analytical solution of the analytical Laplace equation is:
    # u(x) = a + (b - a)/2 * x
    u_analytical = a + (b - a)/2 * X[0]
    der_analytical0 = assemble(derivative((u_analytical**2) * dx, a))
    der_analytical1 = assemble(derivative((u_analytical**2) * dx, b))
    J = assemble(sol * sol * dx)
    J_hat = ReducedFunctional(J, [Control(a), Control(b)])
    adj_derivatives = J_hat.derivative(options={"riesz_representation": "l2"})
    assert np.allclose(adj_derivatives[0].dat.data_ro, der_analytical0.dat.data_ro)
    assert np.allclose(adj_derivatives[1].dat.data_ro, der_analytical1.dat.data_ro)
