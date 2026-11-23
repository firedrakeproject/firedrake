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
    label0, label1 = W._labels

    u, p = TrialFunctions(W)
    if scalar:
        v = TestFunction(V)
        a = inner(u, v)*dx
        idx = slice(None), label0
        other = slice(None), label1
    else:
        q = TestFunction(Q)
        a = inner(p, q)*dx
        idx = slice(None), label1
        other = slice(None), label0

    A = assemble(a, mat_type=mat_type).M

    assert np.allclose(A[idx].values.diagonal(), 0.125)
    assert np.allclose(A[other].values, 0.0)
