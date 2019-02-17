import pytest
import numpy as np
from firedrake import *


@pytest.mark.parametrize("ncell",
                         [1, 4, 10])
def test_conditional(ncell):
    mesh = UnitIntervalMesh(ncell)
    V = FunctionSpace(mesh, "DG", 0)
    u = Function(V)
    du = TrialFunction(V)
    v = TestFunction(V)
    bhp = Constant(2)
    u.dat.data[...] = range(ncell)
    cond = conditional(ge(u-bhp, 0.0), u-bhp, 0.0)
    Fc = cond*v*dx

    A = assemble(derivative(Fc, u, du)).M.values
    expect = np.zeros_like(A)
    for i in range(2, ncell):
        expect[i, i] = 1.0/ncell

    assert np.allclose(A, expect)
