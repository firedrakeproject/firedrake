from os.path import abspath, dirname
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


def test_simple_d2s():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "P", 4)
    f = Function(V).interpolate(Expression("sin(2*pi *(x[0]-x[1]))"))
    assert visualise.simple_d2s(f, 100, 100, 0.01) is None

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
