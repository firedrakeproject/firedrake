from os.path import abspath, dirname, join
import pytest
from firedrake import *
from firedrake.petsc import PETSc

cwd = abspath(dirname(__file__))


@pytest.fixture
def exodus_mesh():
    try:
        return Mesh(join(cwd, "..", "meshes", "brick.e"))
    except PETSc.Error as e:
        assert e.ierr == 56


@pytest.mark.parallel(nprocs=2)
def test_create_from_file(exodus_mesh):
    if exodus_mesh is None:
        pytest.skip("PETSc not configured with exodusII")
    else:
        assert exodus_mesh


@pytest.mark.parallel(nprocs=2)
def test_sidesets(exodus_mesh):
    if exodus_mesh is None:
        pytest.skip("PETSc not configured with exodusII")
    else:
        exodus_mesh.init()
        assert (exodus_mesh.exterior_facets.unique_markers == [200, 201]).all()
