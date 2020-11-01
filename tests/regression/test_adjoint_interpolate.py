import pytest
from numpy.random import rand
from firedrake import *
from pyadjoint.tape import get_working_tape, pause_annotation, annotate_tape


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_exit_annotation():
    yield
    # Since importing firedrake_adjoint modifies a global variable, we need to
    # pause annotations at the end of the module
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_constant():
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = UnitSquareMesh(10, 10)
    V1 = FunctionSpace(mesh, "CG", 1)

    c = Constant(1.0, domain=mesh)
    u = interpolate(c, V1)

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, Control(c))

    h = Constant(0.1, domain=mesh)
    assert taylor_test(rf, c, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_with_arguments():
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = UnitSquareMesh(10, 10)
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)

    x, y = SpatialCoordinate(mesh)
    expr = x + y
    f = interpolate(expr, V1)
    interpolator = Interpolator(TestFunction(V1), V2)
    u = interpolator.interpolate(f)

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V1)
    h.vector()[:] = rand(V1.dim())
    assert taylor_test(rf, f, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_scalar_valued():
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = IntervalMesh(10, 0, 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)
    V3 = FunctionSpace(mesh, "CG", 3)

    x, = SpatialCoordinate(mesh)
    f = interpolate(x, V1)
    g = interpolate(sin(x), V2)
    u = Function(V3)
    u.interpolate(3*f**2 + Constant(4.0)*g)

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
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = UnitSquareMesh(10, 10)
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    V2 = VectorFunctionSpace(mesh, "DG", 0)
    V3 = VectorFunctionSpace(mesh, "CG", 2)

    x = SpatialCoordinate(mesh)
    f = interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V1)
    g = interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V2)
    u = Function(V3)
    u.interpolate(f*dot(f, g) - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V1)
    h.vector()[:] = 1
    assert taylor_test(rf, f, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_tlm():
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = UnitSquareMesh(10, 10)
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    V2 = VectorFunctionSpace(mesh, "DG", 0)
    V3 = VectorFunctionSpace(mesh, "CG", 2)

    x = SpatialCoordinate(mesh)
    f = interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V1)
    g = interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V2)
    u = Function(V3)

    u.interpolate(f - 0.5*g + f/(1+dot(f, g)))
    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V1)
    h.vector()[:] = 1
    f.block_variable.tlm_value = h

    tape = get_working_tape()
    tape.evaluate_tlm()

    assert J.tlm_value is not None
    assert taylor_test(rf, f, h, dJdm=J.tlm_value) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_tlm_wit_constant():
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = IntervalMesh(10, 0, 1)
    V1 = FunctionSpace(mesh, "CG", 2)
    V2 = FunctionSpace(mesh, "DG", 1)

    x = SpatialCoordinate(mesh)
    f = interpolate(x[0], V1)
    g = interpolate(sin(x[0]), V1)
    c = Constant(5.0)
    u = Function(V2)
    u.interpolate(c * f ** 2)

    # test tlm w.r.t constant only:
    c.tlm_value = Constant(1.0)
    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))
    h = Constant(1.0)

    tape = get_working_tape()
    tape.evaluate_tlm()
    assert abs(J.tlm_value - 2.0) < 1e-5
    assert taylor_test(rf, c, h, dJdm=J.tlm_value) > 1.9

    # test tlm w.r.t constant c and function f:
    tape.reset_tlm_values()
    c.tlm_value = Constant(0.4)
    f.block_variable.tlm_value = g
    rf(c)  # replay to reset checkpoint values based on c=5
    tape.evaluate_tlm()
    assert abs(J.tlm_value - (0.8 + 100. * (5*cos(1.) - 3*sin(1.)))) < 1e-4


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_bump_function():
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)

    x, y = SpatialCoordinate(mesh)
    cx = Constant(0.5)
    cy = Constant(0.5)
    f = interpolate(exp(-1/(1-(x-cx)**2)-1/(1-(y-cy)**2)), V)

    J = assemble(f*y**3*dx)
    rf = ReducedFunctional(J, [Control(cx), Control(cy)])

    h = [Constant(0.1), Constant(0.1)]
    assert taylor_test(rf, [cx, cy], h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_self_interpolate():
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

    c = Constant(1.)
    u.interpolate(u+c)

    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))

    h = Constant(0.1)
    assert taylor_test(rf, c, h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_self_interpolate_function():
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)

    c = Constant(1.)
    interpolate(u+c, u)
    interpolate(u+c*u**2, u)

    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))

    h = Constant(0.1)
    assert taylor_test(rf, Constant(3.), h) > 1.9


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_interpolate_to_function_space():
    from firedrake_adjoint import ReducedFunctional, Control, taylor_test
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 1)
    u = Function(V)

    x = SpatialCoordinate(mesh)
    u.interpolate(x[0])
    c = Constant(1.)
    w = interpolate((u+c)*u, W)

    J = assemble(w**2*dx)
    rf = ReducedFunctional(J, Control(c))
    h = Constant(0.1)
    assert taylor_test(rf, Constant(1.), h) > 1.9
