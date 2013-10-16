import pytest

from firedrake import *


@pytest.fixture
def f():
    m = UnitSquareMesh(1, 1)
    fs = FunctionSpace(m, "CG", 1)
    f = Function(fs)
    f.interpolate(Expression("x[0]"))
    return f


def test_external_integral(f):
    assert abs(assemble(f * ds) - 2.0) < 1.0e-14


def test_internal_integral(f):
    assert abs(assemble(f('+') * dS) - 1.0 / (2.0 ** 0.5)) < 1.0e-14


def test_internal_integral_unit_tri():
    t = UnitTriangleMesh()
    V = FunctionSpace(t, 'CG', 1)
    u = Function(V)
    u.interpolate(Expression("x[0]"))
    assert abs(assemble(u('+') * dS) < 1.0e-14)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
