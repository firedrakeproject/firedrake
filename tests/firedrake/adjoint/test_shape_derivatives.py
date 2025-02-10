import pytest

import numpy as np
from firedrake import *
from firedrake.__future__ import *
from firedrake.adjoint import *
from pyadjoint import taylor_to_dict


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
def test_sin_weak_spatial():
    mesh = UnitOctahedralSphereMesh(2)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    S = mesh.coordinates.function_space()
    s = Function(S)
    mesh.coordinates.assign(mesh.coordinates + s)

    J = sin(x[0]) * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()

    V = TestFunction(S)
    # Derivative (Cofunction)
    dJV = assemble(div(V)*sin(x[0])*dx + V[0]*cos(x[0])*dx)
    # Apply L2 riesz representation to obtain the gradient.
    actual = dJV.riesz_representation().vector().get_local()
    assert np.allclose(computed, actual, rtol=1e-14)


@pytest.mark.skipcomplex
def test_tlm_assemble():
    mesh = UnitCubeMesh(4, 4, 4)
    x = SpatialCoordinate(mesh)
    S = mesh.coordinates.function_space()
    h = Function(S)
    A = 10
    h.interpolate(as_vector((A*cos(x[1]), A*x[1], x[2]*x[1])))
    f = Function(S)
    f.interpolate(as_vector((A*sin(x[1]), A*cos(x[1]), sin(x[2]))))
    s = Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)
    J = assemble(sin(x[1]) * dx(domain=mesh))

    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    assert (r0 > 0.95)
    Jhat(s)

    # Tangent linear model
    s.block_variable.tlm_value = h
    tape = get_working_tape()
    tape.evaluate_tlm()
    r1_tlm = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert (r1_tlm > 1.9)
    Jhat(s)
    r1 = taylor_test(Jhat, s, h)
    assert (np.isclose(r1, r1_tlm, rtol=1e-14))


@pytest.mark.skipcomplex
def test_shape_hessian():
    mesh = UnitIcosahedralSphereMesh(3)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
    S = mesh.coordinates.function_space()
    s = Function(S, name="deform")

    mesh.coordinates.assign(mesh.coordinates + s)
    J = assemble(cos(x[1]) * dx(domain=mesh))
    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    A = 10
    h = Function(S, name="V")
    h.interpolate(as_vector((cos(x[2]), A*cos(x[1]), A*x[1])))

    # Second order taylor
    dJdm = assemble(inner(Jhat.derivative(), h)*dx)
    Hm = assemble(inner(compute_hessian(J, c, h), h)*dx)
    r2 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    print(r2)
    assert (r2 > 2.9)
    Jhat(s)
    dJdmm_exact = derivative(derivative(cos(x[1]) * dx(domain=mesh), x, h), x, h)
    assert (np.isclose(assemble(dJdmm_exact), Hm))


@pytest.mark.skipcomplex
def test_PDE_hessian_neumann():
    mesh = UnitOctahedralSphereMesh(refinement_level=2)

    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    S = mesh.coordinates.function_space()
    s = Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)
    f = x[0]*x[1]*x[2]
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx + u*v*dx
    l = f*v*dx
    u = Function(V)
    solve(a == l, u, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu',
                                        "mat_type": "aij",
                                        "pc_factor_mat_solver_type": "mumps"})
    J = assemble(u*dx(domain=mesh))
    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    A = 1e-1
    h = Function(S, name="V")
    h.interpolate(as_vector((A*x[2], A*cos(x[1]), A*x[0])))

    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    Jhat(s)
    assert (r0 > 0.95)

    r1 = taylor_test(Jhat, s, h)
    Jhat(s)
    assert (r1 > 1.95)

    # First order taylor
    s.block_variable.tlm_value = h
    tape = get_working_tape()
    tape.evaluate_tlm()
    r1 = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert (r1 > 1.95)
    Jhat(s)

    # # Second order taylor
    dJdm = assemble(inner(Jhat.derivative(), h)*dx)
    Hm = assemble(inner(compute_hessian(J, c, h), h)*dx)
    r2 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert (r2 > 2.95)


@pytest.mark.skipcomplex
def test_PDE_hessian_dirichlet():

    mesh = UnitTetrahedronMesh()

    x = SpatialCoordinate(mesh)

    S = mesh.coordinates.function_space()
    s = Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)
    f = x[0]*x[1]*x[2]
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    l = f*v*dx
    bc = DirichletBC(V, Constant(1), "on_boundary")
    u = Function(V)
    solve(a == l, u, bc, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu',
                                            "mat_type": "aij",
                                            "pc_factor_mat_solver_type": "mumps"})

    J = assemble(u*dx(domain=mesh))
    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    A = 1e-1
    h = Function(S, name="V")
    h.interpolate(as_vector((A*x[2], A*cos(x[1]), A*x[0])))

    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    Jhat(s)
    assert (r0 > 0.95)

    r1 = taylor_test(Jhat, s, h)
    Jhat(s)
    assert (r1 > 1.95)

    # First order taylor
    s.block_variable.tlm_value = h
    tape = get_working_tape()
    tape.evaluate_tlm()
    r1 = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert (r1 > 1.95)
    Jhat(s)

    # # Second order taylor
    dJdm = assemble(inner(Jhat.derivative(), h)*dx)
    Hm = assemble(inner(compute_hessian(J, c, h), h)*dx)
    r2 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert (r2 > 2.95)


@pytest.mark.skipcomplex
def test_multiple_assignments():

    mesh = UnitSquareMesh(5, 5)
    S = mesh.coordinates.function_space()
    s = Function(S)

    mesh.coordinates.assign(mesh.coordinates + s)
    mesh.coordinates.assign(mesh.coordinates + s)

    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    x, y = SpatialCoordinate(mesh)
    f = (x - 0.2) ** 2 + y ** 2 - 1
    a = dot(grad(u), grad(v)) * dx + u * v * dx
    l = f * v * dx

    u = Function(V)
    solve(a == l, u)
    J = assemble(u * dx)

    Jhat = ReducedFunctional(J, Control(s))
    dJdm = Jhat.derivative()

    pert = as_vector((x * y, sin(x)))
    pert = assemble(interpolate(pert, S))
    results = taylor_to_dict(Jhat, s, pert)

    assert min(results["R0"]["Rate"]) > 0.9
    assert min(results["R1"]["Rate"]) > 1.9
    assert min(results["R2"]["Rate"]) > 2.9

    tape = get_working_tape()
    tape.clear_tape()

    mesh = UnitSquareMesh(5, 5)
    S = mesh.coordinates.function_space()
    s = Function(S)
    mesh.coordinates.assign(mesh.coordinates + 2*s)

    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    x, y = SpatialCoordinate(mesh)
    f = (x - 0.2) ** 2 + y ** 2 - 1
    a = dot(grad(u), grad(v)) * dx + u * v * dx
    l = f * v * dx

    u = Function(V)
    solve(a == l, u)
    J = assemble(u * dx)

    Jhat = ReducedFunctional(J, Control(s))
    assert np.allclose(Jhat.derivative().vector().get_local(),
                       dJdm.vector().get_local())

    pert = as_vector((x * y, sin(x)))
    pert = assemble(interpolate(pert, S))
    results = taylor_to_dict(Jhat, s, pert)

    assert min(results["R0"]["Rate"]) > 0.9
    assert min(results["R1"]["Rate"]) > 1.9
    assert min(results["R2"]["Rate"]) > 2.9
