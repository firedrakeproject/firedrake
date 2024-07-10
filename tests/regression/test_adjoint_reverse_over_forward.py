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

    u = Function(space, name="u").interpolate(X[0])
    zeta = Function(space, name="tlm_u").interpolate(X[0])
    u.block_variable.tlm_value = zeta.copy(deepcopy=True)

    with reverse_over_forward():
        J = assemble(u * u * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(zeta, test) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_constant_assignment():
    a = Constant(2.5)
    a.block_variable.tlm_value = Constant(-2.0)

    with reverse_over_forward():
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

    u = Function(space, name="u").interpolate(X[0] - 0.5)
    zeta = Function(space, name="tlm_u").interpolate(X[0])
    u.block_variable.tlm_value = zeta.copy(deepcopy=True)

    with reverse_over_forward():
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

    u = Function(space, name="u").interpolate(X[0] - 0.5)
    zeta = Function(space, name="tlm_u").interpolate(X[0])
    u.block_variable.tlm_value = zeta.copy(deepcopy=True)

    with reverse_over_forward():
        v = Function(space, name="v").assign(-3 * u)
        J = assemble(v * v * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(18 * inner(zeta, test) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_project():
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_a = FunctionSpace(mesh, "Lagrange", 1)
    space_b = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    test_a = TestFunction(space_a)

    u = Function(space_a, name="u").interpolate(X[0] - 0.5)
    zeta = Function(space_a, name="tlm_u").interpolate(X[0])
    u.block_variable.tlm_value = zeta.copy(deepcopy=True)

    with reverse_over_forward():
        v = Function(space_b, name="v").project(u)
        J = assemble(v * v * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(Function(space_b).project(zeta), test_a) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_supermesh_project():
    mesh_a = UnitSquareMesh(10, 10)
    mesh_b = UnitSquareMesh(5, 20)
    X_a = SpatialCoordinate(mesh_a)
    space_a = FunctionSpace(mesh_a, "Lagrange", 1)
    space_b = FunctionSpace(mesh_b, "Discontinuous Lagrange", 0)
    test_a = TestFunction(space_a)

    u = Function(space_a, name="u").interpolate(X_a[0] - 0.5)
    zeta = Function(space_a, name="tlm_u").interpolate(X_a[0])
    u.block_variable.tlm_value = zeta.copy(deepcopy=True)

    with reverse_over_forward():
        v = Function(space_b, name="v").project(u)
        J = assemble(v * v * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(Function(space_a).project(Function(space_b).project(zeta)), test_a) * dx).dat.data_ro)
