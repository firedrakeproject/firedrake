import pytest
from firedrake import ExtrudedMesh, UnitSquareMesh, UnitIntervalMesh


@pytest.fixture(scope="module")
def extmesh():
    def make_mesh(nx, ny, nz, quadrilateral=False):
        return ExtrudedMesh(UnitSquareMesh(nx, ny, quadrilateral=quadrilateral),
                            nz, layer_height=1.0/nz)
    return make_mesh


@pytest.fixture(scope="module")
def extmesh_2D():
    def make_mesh(nx, ny):
        return ExtrudedMesh(UnitIntervalMesh(nx), ny, layer_height=1.0/ny)
    return make_mesh
