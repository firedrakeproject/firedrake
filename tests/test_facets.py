import pytest
import numpy as np
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


def test_bottom_external_integral(f):
    assert abs(assemble(f * ds(1)) - 0.5) < 1.0e-14


def test_top_external_integral(f):
    assert abs(assemble(f * ds(2)) - 0.5) < 1.0e-14


def test_left_external_integral(f):
    assert abs(assemble(f * ds(4))) < 1.0e-14


def test_right_external_integral(f):
    assert abs(assemble(f * ds(3)) - 1.0) < 1.0e-14


def test_internal_integral(f):
    assert abs(assemble(f('+') * dS) - 1.0 / (2.0 ** 0.5)) < 1.0e-14


def test_facet_integral_with_argument(f):
    v = TestFunction(f.function_space())
    assert np.allclose(assemble(f*v*ds).dat.data_ro.sum(), 2.0)


def test_bilinear_facet_integral(f):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)

    cell = assemble(u*v*dx).M.values
    # each diagonal entry should be volume of cell
    assert np.allclose(np.diag(cell), 0.5)
    # all off-diagonals should be zero
    cell[range(2), range(2)] = 0.0
    assert np.allclose(cell, 0.0)

    outer_facet = assemble(u*v*ds).M.values
    # each diagonal entry should be length of exterior facet in this
    # cell (2)
    assert np.allclose(np.diag(outer_facet), 2.0)
    # all off-diagonals should be zero
    outer_facet[range(2), range(2)] = 0.0
    assert np.allclose(outer_facet, 0.0)

    interior_facet = assemble(u('+')*v('+')*dS).M.values
    # fully coupled, each entry should be length of interior facet
    # (sqrt(2))
    assert np.allclose(interior_facet, sqrt(2))


def test_internal_integral_unit_tri():
    t = UnitTriangleMesh()
    V = FunctionSpace(t, 'CG', 1)
    u = Function(V)
    u.interpolate(Expression("x[0]"))
    assert abs(assemble(u('+') * dS)) < 1.0e-14

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
