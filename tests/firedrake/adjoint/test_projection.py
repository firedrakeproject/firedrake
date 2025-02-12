import pytest

from firedrake import *
from firedrake.__future__ import *
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
def test_project_vector_valued():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V))
    g = assemble(interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V))
    u = Function(V)

    u.project(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(rf, f, h) > 1.9


@pytest.mark.skipcomplex
def test_project_tlm():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V))
    g = assemble(interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V))
    u = Function(V)

    u.project(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = 1
    f.tlm_value = h

    tape = get_working_tape()
    tape.evaluate_tlm()

    assert taylor_test(rf, f, h, dJdm=J.block_variable.tlm_value) > 1.9


@pytest.mark.skipcomplex
def test_project_hessian():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V))
    g = assemble(interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V))
    u = Function(V)

    u.project(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    dJdm = rf.derivative()

    h = Function(V)
    h.vector()[:] = 1.0
    Hm = rf.hessian(h)
    assert taylor_test(rf, f, h, dJdm=h._ad_dot(dJdm), Hm=h._ad_dot(Hm)) > 2.9


@pytest.mark.skipcomplex
def test_project_nonlincom():
    mesh = IntervalMesh(10, 0, 1)
    element = FiniteElement("CG", mesh.ufl_cell(), degree=1)
    V1 = FunctionSpace(mesh, element)
    element = FiniteElement("CG", mesh.ufl_cell(), degree=2)
    V2 = FunctionSpace(mesh, element)
    element = FiniteElement("DG", mesh.ufl_cell(), degree=1)
    V3 = FunctionSpace(mesh, element)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(x[0], V1))
    g = assemble(interpolate(sin(x[0]), V2))
    u = Function(V3)

    u.project(f*g)

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V1)
    h.vector()[:] = rand(V1.dim())
    assert taylor_test(rf, f, h) > 1.9


@pytest.mark.skipcomplex
def test_project_nonlin_changing():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(x[0], V))
    g = assemble(interpolate(sin(x[0]), V))
    control = Control(g)

    test = TestFunction(V)
    trial = TrialFunction(V)
    a = inner(grad(trial), grad(test))*dx
    L = inner(g, test)*dx

    bc = DirichletBC(V, g, "on_boundary")
    sol = Function(V)
    solve(a == L, sol, bc)

    u = Function(V)

    u.project(f*sol*g)

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, control)

    g = Function(V)
    g.vector()[:] = rand(V.dim())

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, g, h) > 1.9


@pytest.mark.skipcomplex
def test_self_project():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = Function(V)
    c = Function(R, val=1.)
    u.project(u+c)
    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))
    assert taylor_test(rf, Function(R, val=2.), Constant(0.1))


@pytest.mark.skipcomplex
def test_self_project_function():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = Function(V)
    c = Function(R, val=1.)
    project(u+c, u)
    project(u+c*u**2, u)
    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(c))
    assert taylor_test(rf, Function(R, val=3.), Constant(0.1))


@pytest.mark.skipcomplex
def test_project_to_function_space():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 1)
    R = FunctionSpace(mesh, "R", 0)
    u = Function(V)
    x = SpatialCoordinate(mesh)
    u.interpolate(x[0])
    c = Function(R, val=1.)
    w = project((u+c)*u, W)
    J = assemble(w**2*dx)
    rf = ReducedFunctional(J, Control(c))
    assert taylor_test(rf, Function(R, val=1.), Constant(0.1)) > 1.9
