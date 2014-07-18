import pytest

from firedrake import *


def test_real_assembly():
    mesh = UnitIntervalMesh(3)
    fs = FunctionSpace(mesh, "Real", 0)
    f = Function(fs)

    f.dat.data[0] = 2.

    assert assemble(f*dx) == 2.0


def test_real_one_form_assembly():
    mesh = UnitIntervalMesh(3)
    fs = FunctionSpace(mesh, "Real", 0)
    v = TestFunction(fs)

    assert assemble(v*dx).dat.data[0] == 1.0


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
