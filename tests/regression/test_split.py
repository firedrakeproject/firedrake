from __future__ import absolute_import, print_function, division
import pytest
from firedrake import *


def test_assemble_split_derivative():
    """Assemble the derivative of a form with a zero block."""
    mesh = UnitSquareMesh(1, 1)
    V1 = FunctionSpace(mesh, "BDM", 1, name="V")
    V2 = FunctionSpace(mesh, "DG", 0, name="P")
    W = V1 * V2

    x = Function(W)
    u, p = split(x)
    v, q = TestFunctions(W)

    F = (inner(u, v) + v[1]*p)*dx

    assert assemble(derivative(F, x))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
