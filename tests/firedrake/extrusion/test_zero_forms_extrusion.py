import numpy as np

from firedrake import *


def test_constant_dx():
    mesh = ExtrudedMesh(UnitIntervalMesh(10), 10)
    one = Constant(1)
    assert np.allclose(assemble(one * dx(domain=mesh)), 1.0)
