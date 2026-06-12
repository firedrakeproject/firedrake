import pytest

from firedrake import *
from firedrake.adjoint import *


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.fixture
def rg():
    return RandomGenerator(PCG64(seed=1234))


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

    h = Function(V).assign(1.)
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

    h = Function(V).assign(1.)
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

    h = Function(V).assign(1.)
    Hm = rf.hessian(h)
    assert taylor_test(rf, f, h, dJdm=h._ad_dot(dJdm), Hm=h._ad_dot(Hm)) > 2.9


@pytest.mark.skipcomplex
def test_project_nonlincom(rg):
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

    h = rg.uniform(V1)
    assert taylor_test(rf, f, h) > 1.9


@pytest.mark.skipcomplex
def test_project_nonlin_changing(rg):
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

    g = rg.uniform(V)

    h = rg.uniform(V)
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
@pytest.mark.parametrize("project_via_method", [False, True])
def test_project_solver_parameters_recorded(project_via_method):
    # A deliberately unconverged solve, so that the forward result is
    # distinguishable from a converged one: the replay only matches the
    # taped functional if it uses the recorded solver parameters.
    sp = {
        "ksp_type": "richardson",
        "pc_type": "none",
        "ksp_max_it": 1,
        "ksp_convergence_test": "skip",
    }

    mesh = UnitSquareMesh(4, 4)
    W = FunctionSpace(mesh, "CG", 2)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(W).interpolate(sin(2 * pi * x) * cos(2 * pi * y))

    if project_via_method:
        u = Function(V).project(f, solver_parameters=sp)
    else:
        u = project(f, V, solver_parameters=sp)
    J = assemble(u**2 * dx)

    from firedrake.adjoint_utils.blocks import ProjectBlock
    block, = (b for b in get_working_tape().get_blocks()
              if isinstance(b, ProjectBlock))
    recorded = block.forward_kwargs["solver_parameters"]
    assert all(recorded[key] == value for key, value in sp.items())
    assert block.adj_kwargs["solver_parameters"] == recorded

    rf = ReducedFunctional(J, Control(f))
    assert rf(f) == pytest.approx(float(J), rel=1e-12)


@pytest.mark.skipcomplex
def test_project_default_solver_parameters_recorded():
    # The Projector's default solver parameters must end up on the tape,
    # so that replay does not fall back to the global firedrake defaults.
    mesh = UnitSquareMesh(4, 4)
    W = FunctionSpace(mesh, "CG", 2)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    f = Function(W).interpolate(x * y)

    u = project(f, V)
    assemble(u**2 * dx)

    from firedrake.adjoint_utils.blocks import ProjectBlock
    from firedrake.projection import resolve_projection_solver_parameters
    block, = (b for b in get_working_tape().get_blocks()
              if isinstance(b, ProjectBlock))
    expected = resolve_projection_solver_parameters(None)
    assert block.forward_kwargs["solver_parameters"] == expected
    assert block.adj_kwargs["solver_parameters"] == expected


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
