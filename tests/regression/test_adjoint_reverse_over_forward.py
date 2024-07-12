from contextlib import contextmanager
import numpy as np
import pytest

from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import *


@pytest.fixture(autouse=True)
def _():
    get_working_tape().clear_tape()
    pause_annotation()
    pause_reverse_over_forward()
    yield
    get_working_tape().clear_tape()
    pause_annotation()
    pause_reverse_over_forward()


@contextmanager
def reverse_over_forward():
    continue_annotation()
    continue_reverse_over_forward()
    yield
    pause_annotation()
    pause_reverse_over_forward()


@pytest.mark.skipcomplex
def test_assembly():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    with reverse_over_forward():
        u = Function(space, name="u").interpolate(X[0])
        zeta = Function(space, name="zeta").interpolate(X[0])
        u.block_variable.tlm_value = zeta.copy(deepcopy=True)

        J = assemble(u * u * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(zeta, test) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_constant_assignment():
    with reverse_over_forward():
        a = Constant(2.5)
        a.block_variable.tlm_value = Constant(-2.0)

        b = Constant(0.0).assign(a)

    assert float(b.block_variable.tlm_value) == -2.0

    # Minimal test that the TLM operation is on the tape
    _ = compute_gradient(b.block_variable.tlm_value, Control(a.block_variable.tlm_value))
    adj_value = a.block_variable.tlm_value.block_variable.adj_value
    assert float(adj_value) == 1.0


@pytest.mark.skipcomplex
def test_function_assignment():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    with reverse_over_forward():
        u = Function(space, name="u").interpolate(X[0] - 0.5)
        zeta = Function(space, name="zeta").interpolate(X[0])
        u.block_variable.tlm_value = zeta.copy(deepcopy=True)

        v = Function(space, name="v").assign(u)
        J = assemble(v * v * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(zeta, test) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_function_assignment_expr():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    with reverse_over_forward():
        u = Function(space, name="u").interpolate(X[0] - 0.5)
        zeta = Function(space, name="zeta").interpolate(X[0])
        u.block_variable.tlm_value = zeta.copy(deepcopy=True)

        v = Function(space, name="v").assign(-3 * u)
        J = assemble(v * v * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(18 * inner(zeta, test) * dx).dat.data_ro)


@pytest.mark.skipcomplex
@pytest.mark.parametrize("idx", [0, 1])
def test_subfunction(idx):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1) * FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    with reverse_over_forward():
        u = Function(space, name="u")
        u.sub(idx).interpolate(-2 * X[0])
        zeta = Function(space, name="zeta")
        zeta.sub(idx).interpolate(X[0])
        u.block_variable.tlm_value = zeta.copy(deepcopy=True)

        v = Function(space, name="v")
        v.sub(idx).assign(u.sub(idx))
        J = assemble(u.sub(idx) * v.sub(idx) * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.sub(idx).dat.data_ro,
                       assemble(2 * inner(zeta[idx], test[idx]) * dx).dat.data_ro[idx])


@pytest.mark.skipcomplex
def test_interpolate():
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_a = FunctionSpace(mesh, "Lagrange", 1)
    space_b = FunctionSpace(mesh, "Lagrange", 2)
    test_a = TestFunction(space_a)

    with reverse_over_forward():
        u = Function(space_a, name="u").interpolate(X[0] - 0.5)
        zeta = Function(space_a, name="zeta").interpolate(X[0])
        u.block_variable.tlm_value = zeta.copy(deepcopy=True)

        v = Function(space_b, name="v").interpolate(u)
        J = assemble(v * v * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(zeta, test_a) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_project():
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_a = FunctionSpace(mesh, "Lagrange", 1)
    space_b = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    test_a = TestFunction(space_a)

    with reverse_over_forward():
        u = Function(space_a, name="u").interpolate(X[0] - 0.5)
        zeta = Function(space_a, name="zeta").interpolate(X[0])
        u.block_variable.tlm_value = zeta.copy(deepcopy=True)

        v = Function(space_b, name="v").project(u)
        J = assemble(v * v * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(Function(space_b).project(zeta), test_a) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_project_overwrite():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    with reverse_over_forward():
        u = Function(space, name="u").interpolate(X[0] - 0.5)
        zeta = Function(space, name="zeta").interpolate(X[0])
        u.block_variable.tlm_value = zeta.copy(deepcopy=True)

        v = Function(space).project(-2 * u)
        w = Function(space).assign(u)
        w.project(-2 * w)
        assert np.allclose(w.dat.data_ro, v.dat.data_ro)
        assert np.allclose(v.block_variable.tlm_value.dat.data_ro,
                           -2 * zeta.dat.data_ro)
        assert np.allclose(w.block_variable.tlm_value.dat.data_ro,
                           -2 * zeta.dat.data_ro)
        J = assemble(w * w * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(8 * inner(zeta, test) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_supermesh_project():
    mesh_a = UnitSquareMesh(10, 10)
    mesh_b = UnitSquareMesh(5, 20)
    X_a = SpatialCoordinate(mesh_a)
    space_a = FunctionSpace(mesh_a, "Lagrange", 1)
    space_b = FunctionSpace(mesh_b, "Discontinuous Lagrange", 0)
    test_a = TestFunction(space_a)

    with reverse_over_forward():
        u = Function(space_a, name="u").interpolate(X_a[0] - 0.5)
        zeta = Function(space_a, name="zeta").interpolate(X_a[0])
        u.block_variable.tlm_value = zeta.copy(deepcopy=True)

        v = Function(space_b, name="v").project(u)
        J = assemble(v * v * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(Function(space_a).project(Function(space_b).project(zeta)), test_a) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_dirichletbc():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    with reverse_over_forward():
        u = Function(space, name="u").interpolate(X[0] - 0.5)
        zeta = Function(space, name="zeta").interpolate(X[0])
        u.block_variable.tlm_value = zeta.copy(deepcopy=True)
        bc = DirichletBC(space, u, "on_boundary")

        v = project(Constant(0.0), space, bcs=bc)
        J = assemble(v * v * v * dx)

    J_hat = ReducedFunctional(J, Control(u))
    assert taylor_test(J_hat, u, zeta, dJdm=J.block_variable.tlm_value) > 1.9
    J_hat = ReducedFunctional(J.block_variable.tlm_value, Control(u))
    assert taylor_test(J_hat, u, Function(space).interpolate(X[0] * X[0])) > 1.9
