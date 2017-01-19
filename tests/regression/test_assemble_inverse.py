from __future__ import absolute_import, print_function, division
import pytest
import numpy as np
from firedrake import *


@pytest.mark.parametrize("degree", range(4))
def test_assemble_inverse(degree):
    m = UnitSquareMesh(2, 1)
    fs = FunctionSpace(m, "DG", degree)
    u = TrialFunction(fs)
    v = TestFunction(fs)

    m_forward = assemble(u*v*dx)
    m_inverse = assemble(u*v*dx, inverse=True)

    eye = np.dot(m_forward.M.values, m_inverse.M.values)

    assert ((eye - np.eye(fs.node_count)).round(12) == 0).all()
