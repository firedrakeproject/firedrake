import numpy as np
import pytest

from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import *


@pytest.fixture(autouse=True, scope="module")
def _():
    get_working_tape().clear_tape()
    pause_annotation()
    pause_reverse_over_forward()
    yield
    get_working_tape().clear_tape()
    pause_annotation()
    pause_reverse_over_forward()


def test_assembly():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    u = Function(space, name="u").interpolate(Constant(1.0))
    zeta = Function(space, name="tlm_u").interpolate(X[0])
    u.block_variable.tlm_value = zeta.copy(deepcopy=True)

    continue_annotation()
    continue_reverse_over_forward()
    J = assemble(u * u * dx)
    pause_annotation()
    pause_reverse_over_forward()

    _ = compute_gradient(J.block_variable.tlm_value, Control(u))
    adj_value = u.block_variable.adj_value
    assert np.allclose(adj_value.dat.data_ro,
                       assemble(2 * inner(zeta, test) * dx).dat.data_ro)
