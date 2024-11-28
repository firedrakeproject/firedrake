from firedrake import *
import pytest
import numpy as np


@pytest.fixture
def mesh():
    return UnitSquareMesh(2, 2)


@pytest.fixture
def V(mesh):
    return FunctionSpace(mesh, "DG", 0)


@pytest.fixture
def Q(mesh):
    return VectorFunctionSpace(mesh, "DG", 0)


@pytest.mark.parametrize("mat_type", ["nest", "aij"])
@pytest.mark.parametrize("scalar", [False, True],
                         ids=["Vector", "Scalar"])
def test_partially_mixed_mat(V, Q, mat_type, scalar):

    W = V*Q

    u, p = TrialFunctions(W)
    if scalar:
        v = TestFunction(V)
        a = inner(u, v)*dx
        idx = 0, 0
        other = 0, 1
    else:
        q = TestFunction(Q)
        a = inner(p, q)*dx
        idx = 0, 1
        other = 0, 0

    A = assemble(a, mat_type=mat_type).M

    assert np.allclose(A[idx].values.diagonal(), 0.125)
    assert np.allclose(A[other].values, 0.0)
