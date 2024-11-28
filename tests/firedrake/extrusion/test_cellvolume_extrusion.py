import pytest
import numpy as np
from firedrake import *


@pytest.fixture(params=["interval", "triangle", "quadrilateral"])
def basecell(request):
    return request.param


@pytest.fixture
def mesh(basecell):
    if basecell == "interval":
        basemesh = UnitIntervalMesh(1)
    if basecell == "triangle":
        basemesh = UnitTriangleMesh()
    if basecell == "quadrilateral":
        basemesh = UnitSquareMesh(1, 1, quadrilateral=True)
    return ExtrudedMesh(basemesh, 1)


@pytest.fixture
def expect(basecell):
    return {"interval": 1.0,
            "triangle": 1.0/2.0,
            "quadrilateral": 1.0}[basecell]


@pytest.mark.parametrize("exponent",
                         [1, 0.5])
def test_cell_volume(exponent, mesh, expect):
    assert np.allclose(assemble((CellVolume(mesh)**exponent)*dx), expect**(exponent + 1))


@pytest.mark.parametrize("measure", [ds_b, ds_t, ds_v])
def test_cell_volume_exterior_facet(mesh, expect, measure):
    assert np.allclose(assemble(sqrt(CellVolume(mesh))*measure),
                       assemble(1 * measure(domain=mesh)) * sqrt(expect))


def test_facet_area(basecell, mesh):
    expect = {"interval": 4.0,
              "triangle": 4.5,
              "quadrilateral": 6.0}[basecell]
    assert np.allclose(assemble(FacetArea(mesh)*(ds_b + ds_t + ds_v)), expect)


def test_miscellaneous():
    mesh = ExtrudedMesh(UnitSquareMesh(1, 1), 2)
    mesh.coordinates.dat.data[:, 2] = np.sqrt(mesh.coordinates.dat.data_ro[:, 2])

    assert np.allclose(assemble(CellVolume(mesh)*dx), 1 - 1/sqrt(2))
    assert np.allclose(assemble(CellVolume(mesh)*ds_b), sqrt(2)/4)
    assert np.allclose(assemble(CellVolume(mesh)*ds_t), 0.5 - sqrt(2)/4)
    assert np.allclose(assemble(CellVolume(mesh)*ds_v), 4 - 2*sqrt(2))
    assert np.allclose(sorted([assemble(CellVolume(mesh)('+')*dS_h),
                               assemble(CellVolume(mesh)('-')*dS_h)]),
                       [(2 - sqrt(2))/4, sqrt(2)/4])
    assert np.allclose(assemble(CellVolume(mesh)('+')*dS_v), sqrt(2) - 1)
    assert np.allclose(assemble(CellVolume(mesh)('-')*dS_v), sqrt(2) - 1)

    with pytest.raises(ValueError):
        assemble(FacetArea(mesh)*dx)

    assert np.allclose(assemble(FacetArea(mesh)*ds_b), 0.5)
    assert np.allclose(assemble(FacetArea(mesh)*dS_h), 0.5)
    assert np.allclose(assemble(FacetArea(mesh)*ds_t), 0.5)
    assert np.allclose(assemble(FacetArea(mesh)*ds_v), 4*(2 - sqrt(2)))
    assert np.allclose(assemble(FacetArea(mesh)*dS_v), 2*(2 - sqrt(2)))
