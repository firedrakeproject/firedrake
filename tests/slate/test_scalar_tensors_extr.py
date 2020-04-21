import numpy as np
import pytest

from firedrake import *


@pytest.mark.skipif(utils.complex_mode, reason="Slate does not work for complex.")
def test_constant_one_tensor():
    mesh = ExtrudedMesh(UnitIntervalMesh(5), 5)
    one = Constant(1, domain=mesh)
    assert np.allclose(assemble(Tensor(one * dx)), 1.0)
