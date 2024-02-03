import pytest
import numpy as np
from ufl.geometry import CellOrigin
from firedrake import *
from firedrake.__future__ import *


@pytest.fixture(params=["interval", "triangle", "quadrilateral", "tetrahedron"])
def cell(request):
    return request.param


@pytest.fixture
def mesh(cell):
    if cell == "interval":
        return UnitIntervalMesh(10)
    if cell == "triangle":
        return UnitSquareMesh(5, 5)
    if cell == "quadrilateral":
        return UnitSquareMesh(5, 5, quadrilateral=True)
    if cell == "tetrahedron":
        return UnitCubeMesh(2, 2, 2)


def test_cell_origin(mesh):
    V = VectorFunctionSpace(mesh, "DG", 0)
    f = assemble(interpolate(CellOrigin(mesh), V))

    coords = mesh.coordinates
    expected = coords.dat.data_ro[coords.function_space().cell_node_list[:, 0]]
    assert np.allclose(expected, f.dat.data_ro)
