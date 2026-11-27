from os.path import abspath, dirname, join
import pytest
from firedrake import *
from petsctools import get_external_packages

cwd = abspath(dirname(__file__))


@pytest.fixture
def exodus_mesh() -> MeshGeometry:
    if "exodusii" not in get_external_packages():
        pytest.skip("PETSc was built without ExodusII support")
    return Mesh(join(cwd, "..", "meshes", "brick.e"))


@pytest.mark.parallel(nprocs=2)
def test_create_from_file(exodus_mesh):
    assert exodus_mesh


@pytest.mark.parallel([1, 2])
def test_sidesets(exodus_mesh: MeshGeometry) -> None:
    assert (exodus_mesh.facet_markers == [200, 201]).all()
    facet_names = exodus_mesh.topology.region_names[exodus_mesh.topological_dimension - 1]
    boundary = ds(domain=exodus_mesh)
    for name, marker in (("Sideset_A", 200), ("Sideset_B", 201)):
        assert facet_names[name] == [marker]
        assert exodus_mesh.topology_dm.hasLabel(name)
        numeric_area = assemble(1 * boundary(marker))
        assert abs(numeric_area) > 0
        assert assemble(1 * boundary(name)) == pytest.approx(numeric_area)
