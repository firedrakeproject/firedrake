import pytest
from firedrake import *


@pytest.fixture
def exodus_mesh():
    return Mesh("../meshes/brick.e")


@pytest.mark.parallel(nprocs=2)
def test_create_from_file(exodus_mesh):
    assert exodus_mesh


@pytest.mark.parallel(nprocs=2)
def test_sidesets(exodus_mesh):
    exodus_mesh.init()
    assert (exodus_mesh.exterior_facets.unique_markers == [200, 201]).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
