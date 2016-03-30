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


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
