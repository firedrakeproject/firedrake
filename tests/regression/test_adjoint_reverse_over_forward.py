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

    u = Function(space, name="u").interpolate(Constant(1.0))
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

    # Minimal test that the TLM operations are on the tape
    _ = compute_gradient(b.block_variable.tlm_value, Control(a.block_variable.tlm_value))
    adj_value = a.block_variable.tlm_value.block_variable.adj_value
    assert float(adj_value) == 1.0
