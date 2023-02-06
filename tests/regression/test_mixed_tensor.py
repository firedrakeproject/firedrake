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

    V, Q, T = W.subfunctions

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

    assert np.allclose(A00, A[0, 0].values)
    assert np.allclose(A11, A[1, 1].values)
    assert np.allclose(A22, A[2, 2].values)

    for i in range(3):
        for j in range(3):
            if i != j:
                assert np.allclose(A[i, j].values, 0)
