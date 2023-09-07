import pytest
import numpy as np
from firedrake import *


@pytest.fixture(params=[False, True])
def f(request):
    quadrilateral = request.param
    m = UnitSquareMesh(1, 1, quadrilateral=quadrilateral)
    fs = FunctionSpace(m, "CG", 1)
    f = Function(fs)
    x = SpatialCoordinate(m)
    f.interpolate(x[0])
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
    if f.function_space().mesh().num_cells() == 1:
        # Quadrilateral case, no internal facet
        assert abs(assemble(f('+') * dS)) < 1.0e-14
    else:
        # Triangle case, one internal facet
        assert abs(assemble(f('+') * dS) - 1.0 / (2.0 ** 0.5)) < 1.0e-14


def test_facet_integral_with_argument(f):
    v = TestFunction(f.function_space())
    assert np.allclose(assemble(inner(f, v) * ds).dat.data_ro.sum(), 2.0)


def test_bilinear_cell_integral(dg_trial_test):
    u, v = dg_trial_test
    cell = assemble(inner(u, v) * dx).M.values
    # each diagonal entry should be volume of cell
    assert np.allclose(np.diag(cell), 0.5)
    # all off-diagonals should be zero
    cell[range(2), range(2)] = 0.0
    assert np.allclose(cell, 0.0)


def test_bilinear_exterior_facet_integral(dg_trial_test):
    u, v = dg_trial_test
    outer_facet = assemble(inner(u, v) * ds).M.values
    # each diagonal entry should be length of exterior facet in this
    # cell (2)
    assert np.allclose(np.diag(outer_facet), 2.0)
    # all off-diagonals should be zero
    outer_facet[range(2), range(2)] = 0.0
    assert np.allclose(outer_facet, 0.0)


def test_vector_bilinear_exterior_facet_integral():
    mesh = IntervalMesh(5, 5)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * ds
    A = assemble(a)
    values = A.M.values

    # Only the first and last vertices should contain nonzeros. Since these are
    # blocked that means that the first two entries and the last two entries
    # should be nonzero.
    nonzeros = [[0, 0], [1, 1], [-2, -2], [-1, -1]]
    assert all(np.allclose(values[row, col], 1.0) for row, col in nonzeros)

    # the remaining entries should all be zero
    for row, col in nonzeros:
        values[row, col] = 0.0
    assert np.allclose(values, 0.0)


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
        form = form + inner(u(u_r), v(v_r)) * dS
        exact[idx[v_r], idx[u_r]] += sqrt(2)

    interior_facet = assemble(form).M.values

    assert np.allclose(interior_facet - exact, 0.0)


@pytest.mark.parametrize('space', ["RT", "BDM"])
def test_contravariant_piola_facet_integral(space):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, space, 1)
    u = project(Constant((0.0, 1.0)), V)
    assert abs(assemble(inner(u('+'), u('+'))*dS) - sqrt(2)) < 1.0e-13
    assert abs(assemble(inner(u('-'), u('-'))*dS) - sqrt(2)) < 1.0e-13
    assert abs(assemble(inner(u('+'), u('-'))*dS) - sqrt(2)) < 1.0e-13


@pytest.mark.parametrize('space', ["N1curl", "N2curl"])
def test_covariant_piola_facet_integral(space):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, space, 1)
    u = project(Constant((0.0, 1.0)), V)
    assert abs(assemble(inner(u('+'), u('+'))*dS) - sqrt(2)) < 1.0e-13
    assert abs(assemble(inner(u('-'), u('-'))*dS) - sqrt(2)) < 1.0e-13
    assert abs(assemble(inner(u('+'), u('-'))*dS) - sqrt(2)) < 1.0e-13


def test_internal_integral_unit_tri():
    t = UnitTriangleMesh()
    V = FunctionSpace(t, 'CG', 1)
    u = Function(V)
    x = SpatialCoordinate(t)
    u.interpolate(x[0])
    assert abs(assemble(u('+') * dS)) < 1.0e-14


def test_internal_integral_unit_tet():
    t = UnitTetrahedronMesh()
    V = FunctionSpace(t, 'CG', 1)
    u = Function(V)
    x = SpatialCoordinate(t)
    u.interpolate(x[0])
    assert abs(assemble(u('+') * dS)) < 1.0e-14


def test_facet_map_no_reshape():
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, "DG", 0)
    efnm = V.exterior_facet_node_map()
    assert efnm.values_with_halo.shape == (4, 1)


def test_mesh_with_no_facet_markers():
    mesh = UnitTriangleMesh()
    mesh.init()
    with pytest.raises(LookupError):
        mesh.exterior_facets.subset((10,))
