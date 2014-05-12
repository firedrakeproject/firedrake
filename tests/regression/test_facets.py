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


@pytest.fixture(scope='module')
def dg_trial_test():
    # Interior facet tests hard code order in which cells were
    # numbered, so don't reorder this mesh.
    m = UnitSquareMesh(1, 1, reorder=False)
    V = FunctionSpace(m, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    return u, v


def test_external_integral(f):
    assert abs(assemble(f * ds) - 2.0) < 1.0e-14


def test_bottom_external_integral(f):
    assert abs(assemble(f * ds(3)) - 0.5) < 1.0e-14


def test_top_external_integral(f):
    assert abs(assemble(f * ds(4)) - 0.5) < 1.0e-14


def test_left_external_integral(f):
    assert abs(assemble(f * ds(1))) < 1.0e-14


def test_right_external_integral(f):
    assert abs(assemble(f * ds(2)) - 1.0) < 1.0e-14


def test_internal_integral(f):
    assert abs(assemble(f('+') * dS) - 1.0 / (2.0 ** 0.5)) < 1.0e-14


def test_facet_integral_with_argument(f):
    v = TestFunction(f.function_space())
    assert np.allclose(assemble(f*v*ds).dat.data_ro.sum(), 2.0)


def test_bilinear_cell_integral(dg_trial_test):
    u, v = dg_trial_test
    cell = assemble(u*v*dx).M.values
    # each diagonal entry should be volume of cell
    assert np.allclose(np.diag(cell), 0.5)
    # all off-diagonals should be zero
    cell[range(2), range(2)] = 0.0
    assert np.allclose(cell, 0.0)


def test_bilinear_exterior_facet_integral(dg_trial_test):
    u, v = dg_trial_test
    outer_facet = assemble(u*v*ds).M.values
    # each diagonal entry should be length of exterior facet in this
    # cell (2)
    assert np.allclose(np.diag(outer_facet), 2.0)
    # all off-diagonals should be zero
    outer_facet[range(2), range(2)] = 0.0
    assert np.allclose(outer_facet, 0.0)


@pytest.mark.parametrize('restrictions',
                         # ((trial space restrictions), (test space restrictions))
                         [(('+', ), ('+', )),
                          (('+', ), ('-', )),
                          (('-', ), ('+', )),
                          (('-', '+'), ('+', '+')),
                          (('-', '+'), ('-', '+')),
                          (('-', '+'), ('+', '-')),
                          (('-', '+'), ('-', '-')),
                          (('+', '+'), ('+', '+')),
                          (('+', '+'), ('-', '+')),
                          (('+', '+'), ('+', '-')),
                          (('+', '+'), ('-', '-')),
                          (('-', '-'), ('+', '+')),
                          (('-', '-'), ('-', '+')),
                          (('-', '-'), ('+', '-')),
                          (('-', '-'), ('-', '-')),
                          (('+', '-'), ('+', '+')),
                          (('+', '-'), ('-', '+')),
                          (('+', '-'), ('+', '-')),
                          (('+', '-'), ('-', '-')),
                          (('+', '+', '-', '-'), ('+', '-', '+', '-'))])
def test_bilinear_interior_facet_integral(dg_trial_test, restrictions):
    u, v = dg_trial_test
    trial_r, test_r = restrictions

    idx = {'+': 0, '-': 1}
    exact = np.zeros((2, 2), dtype=float)

    form = 0
    for u_r, v_r in zip(trial_r, test_r):
        form = form + u(u_r)*v(v_r)*dS
        exact[idx[v_r], idx[u_r]] += sqrt(2)

    interior_facet = assemble(form).M.values

    assert np.allclose(interior_facet - exact, 0.0)


@pytest.mark.parametrize('space', ["RT", "BDM"])
def test_contravariant_piola_facet_integral(space):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, space, 1)
    u = project(Expression(("0.0", "1.0")), V)
    assert abs(assemble(dot(u('+'), u('+'))*dS) - sqrt(2)) < 1.0e-13
    assert abs(assemble(dot(u('-'), u('-'))*dS) - sqrt(2)) < 1.0e-13
    assert abs(assemble(dot(u('+'), u('-'))*dS) - sqrt(2)) < 1.0e-13


@pytest.mark.parametrize('space', ["N1curl", "N2curl"])
def test_covariant_piola_facet_integral(space):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, space, 1)
    u = project(Expression(("0.0", "1.0")), V)
    assert abs(assemble(dot(u('+'), u('+'))*dS) - sqrt(2)) < 1.0e-13
    assert abs(assemble(dot(u('-'), u('-'))*dS) - sqrt(2)) < 1.0e-13
    assert abs(assemble(dot(u('+'), u('-'))*dS) - sqrt(2)) < 1.0e-13


def test_internal_integral_unit_tri():
    t = UnitTriangleMesh()
    V = FunctionSpace(t, 'CG', 1)
    u = Function(V)
    u.interpolate(Expression("x[0]"))
    assert abs(assemble(u('+') * dS)) < 1.0e-14


def test_internal_integral_unit_tet():
    t = UnitTetrahedronMesh()
    V = FunctionSpace(t, 'CG', 1)
    u = Function(V)
    u.interpolate(Expression("x[0]"))
    assert abs(assemble(u('+') * dS)) < 1.0e-14


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
