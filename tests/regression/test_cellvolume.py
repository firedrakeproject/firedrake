import pytest
import numpy as np
from firedrake import *


@pytest.fixture(params=["interval", "triangle", "quadrilateral", "tetrahedron"])
def cell(request):
    return request.param


@pytest.fixture
def mesh(cell):
    if cell == "interval":
        return UnitIntervalMesh(1)
    if cell == "triangle":
        return UnitTriangleMesh()
    if cell == "quadrilateral":
        return UnitSquareMesh(1, 1, quadrilateral=True)
    if cell == "tetrahedron":
        return UnitTetrahedronMesh()


@pytest.fixture
def expect(cell):
    return {"interval": 1.0,
            "triangle": 1.0/2.0,
            "quadrilateral": 1.0,
            "tetrahedron": 1.0/6.0}[cell]


@pytest.mark.parametrize("exponent",
                         [1, 0.5])
def test_cell_volume(exponent, mesh, expect):
    assert np.allclose(assemble((CellVolume(mesh)**exponent)*dx), expect**(exponent + 1))


def test_cell_volume_exterior_facet(mesh, expect):
    assert np.allclose(assemble(sqrt(CellVolume(mesh))*ds),
                       assemble(1 * ds(domain=mesh)) * sqrt(expect))


def test_facet_area(cell, mesh):
    expect = {"interval": 2.0,
              "triangle": 4.0,
              "quadrilateral": 4.0,
              "tetrahedron": 1.5}[cell]
    assert np.allclose(assemble(FacetArea(mesh)*ds), expect)


def test_miscellaneous():
    mesh = UnitSquareMesh(2, 1, quadrilateral=True)
    mesh.coordinates.dat.data[:, 0] = np.sqrt(mesh.coordinates.dat.data_ro[:, 0])

    assert np.allclose(assemble(CellVolume(mesh)*dx), 2 - sqrt(2))
    assert np.allclose(assemble(CellVolume(mesh)*ds), 5 - 2*sqrt(2))
    assert np.allclose(sorted([assemble(CellVolume(mesh)('+')*dS),
                               assemble(CellVolume(mesh)('-')*dS)]),
                       [1 - 1/sqrt(2), 1/sqrt(2)])

    with pytest.raises(ValueError):
        assemble(FacetArea(mesh)*dx)

    assert np.allclose(assemble(FacetArea(mesh)*ds), 2*(3 - sqrt(2)))
    assert np.allclose(assemble(FacetArea(mesh)*dS), 1)
