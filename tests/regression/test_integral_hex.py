import pytest
from firedrake import *
from os.path import abspath, dirname, join


cwd = abspath(dirname(__file__))


# @pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('mesh_from_file', [False, True])
@pytest.mark.parametrize('family', ["Q", "DQ"])
def test_integral_hex_exterior_facet(mesh_from_file, family):
    # FIXME: Cope with reorder=True
    if mesh_from_file:
        # mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"))
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"), reorder=False)
    else:
        # mesh = UnitCubeMesh(2, 3, 5, hexahedral=True)
        mesh = UnitCubeMesh(2, 3, 5, hexahedral=True, reorder=False)

    # breakpoint()
    V = FunctionSpace(mesh, family, 3)
    x, y, z = SpatialCoordinate(mesh)
    f = Function(V).interpolate(2 * x + 3 * y * y + 4 * z * z * z)
    assert abs(assemble(f * ds) - (2 + 4 + 2 + 5 + 2 + 6)) < 1.e-10
