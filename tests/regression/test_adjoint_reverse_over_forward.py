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
        J = assemble(u * u.dx(0) * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(inner(zeta.dx(0), test) * dx + inner(zeta, test.dx(0)) * dx).dat.data_ro)


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
        v = Function(space, name="v").assign(-3 * u)
        J = assemble(v * v.dx(0) * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(9 * inner(zeta.dx(0), test) * dx + 9 * inner(zeta, test.dx(0)) * dx).dat.data_ro)


@pytest.mark.skipcomplex
def test_project():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    u = Function(space, name="u").interpolate(X[0] - 0.5)
    zeta = Function(space, name="tlm_u").interpolate(X[0])
    u.block_variable.tlm_value = zeta.copy(deepcopy=True)

    space_0 = FunctionSpace(mesh, "Discontinuous Lagrange", 0)

    with reverse_over_forward():
        v = Function(space_0, name="v").project(u)
        J = assemble(v * v * dx)

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(Function(space_0).project(zeta), test) * dx).dat.data_ro)
