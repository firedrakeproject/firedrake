import numpy as np
from math import ceil

from firedrake import *


def test_constant_one_tensor():
    mesh = ExtrudedMesh(UnitIntervalMesh(5), 5)
    one = Constant(1)
    assert np.allclose(assemble(Tensor(one * dx(domain=mesh))), 1.0)
