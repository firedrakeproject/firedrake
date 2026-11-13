from firedrake import *
import numpy as np
import pytest


def test_single_mesh_mixed_assign():
    """Assigning between functions on separately constructed but equivalent
    MixedFunctionSpaces should work and preserve values."""
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "CG", 1)

    z = Function(MixedFunctionSpace([V, W]))
    z.subfunctions[0].assign(Constant((1.0, 2.0)))
    z.subfunctions[1].assign(3.0)

    w = Function(MixedFunctionSpace([V, W]))
    w.assign(z)

    assert np.allclose(w.subfunctions[0].dat.data_ro, [1.0, 2.0])
    assert np.allclose(w.subfunctions[1].dat.data_ro, 3.0)


@pytest.mark.parallel([1, 3])
def test_multiple_mesh_self_assign():
    mesh1 = UnitSquareMesh(2, 2)
    mesh2 = UnitSquareMesh(3, 3)
    V1 = FunctionSpace(mesh1, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)
    W = V1 * V2

    u = Function(W)
    u.subfunctions[0].assign(1.0)
    u.subfunctions[1].assign(2.0)

    u.assign(2 * u)

    assert np.allclose(u.subfunctions[0].dat.data_ro, 2.0)
    assert np.allclose(u.subfunctions[1].dat.data_ro, 4.0)
