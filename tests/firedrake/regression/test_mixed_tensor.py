import pytest
from firedrake import *
import numpy as np


@pytest.fixture(scope="module")
def W():
    m = UnitSquareMesh(1, 1)

    V = FunctionSpace(m, "DG", 1)
    Q = TensorFunctionSpace(m, "CG", 1)
    T = TensorFunctionSpace(m, "DG", 0, shape=(2, 3))

    return V*Q*T


def test_mass_mixed_tensor(W):

    u, p, s = TrialFunctions(W)
    v, q, t = TestFunctions(W)

    a = (inner(u, v) + inner(p, q) + inner(s, t))*dx

    V, Q, T = W.subspaces
    label0, label1, label2 = W._labels

    u = TrialFunction(V)
    v = TestFunction(V)

    p = TrialFunction(Q)
    q = TestFunction(Q)

    s = TrialFunction(T)
    t = TestFunction(T)

    A00 = assemble(inner(u, v)*dx).M.values
    A11 = assemble(inner(p, q)*dx).M.values
    A22 = assemble(inner(s, t)*dx).M.values

    A = assemble(a).M

    assert np.allclose(A00, A[label0, label0].values)
    assert np.allclose(A11, A[label1, label1].values)
    assert np.allclose(A22, A[label2, label2].values)

    for label_i in W._labels:
        for label_j in W._labels:
            if label_i != label_j:
                assert np.allclose(A[label_i, label_j].values, 0)
