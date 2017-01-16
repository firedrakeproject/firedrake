from __future__ import absolute_import, print_function, division
import pytest
import numpy as np

from firedrake import *


def test_constant_dx():
    mesh = ExtrudedMesh(UnitIntervalMesh(10), 10)
    one = Constant(1, domain=mesh)
    assert np.allclose(assemble(one * dx), 1.0)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
