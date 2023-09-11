import pytest
import numpy as np
from firedrake import *
from ufl.geometry import CellCoordinate, FacetCoordinate


@pytest.fixture(scope='module')
def mesh():
    return UnitTriangleMesh()


def test_cell_coordinate_dx(mesh):
    assert np.allclose(1.0 / 6, assemble(CellCoordinate(mesh)[0]*dx))


def test_cell_coordinate_ds(mesh):
    assert np.allclose(0.5 * (1 + sqrt(2)), assemble(CellCoordinate(mesh)[0]*ds))


def test_cell_coordinate_dS_not_restricted():
    mesh = UnitSquareMesh(1, 1)
    with pytest.raises(ValueError):
        assemble(CellCoordinate(mesh)[0]*dS)


def test_cell_coordinate_dS():
    mesh = UnitSquareMesh(1, 1)
    expected = {0.0, 0.5, 0.5 * np.sqrt(2)}

    actual = assemble(CellCoordinate(mesh)('+')[0]*dS)
    assert any(np.allclose(x, actual) for x in expected)

    actual = assemble(CellCoordinate(mesh)('-')[0]*dS)
    assert any(np.allclose(x, actual) for x in expected)


def test_facet_coordinate_dx(mesh):
    with pytest.raises(ValueError):
        assemble(FacetCoordinate(mesh)[0]*dx)


def test_facet_coordinate_ds(mesh):
    assert np.allclose(0.5 * (2 + sqrt(2)), assemble(FacetCoordinate(mesh)[0]*ds))
