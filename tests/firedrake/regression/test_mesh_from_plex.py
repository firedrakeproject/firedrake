from firedrake import *
import numpy as np


def remove_pyop2_label(plex):
    plex.removeLabel("pyop2_core")
    plex.removeLabel("pyop2_owned")
    plex.removeLabel("pyop2_ghost")
    return plex


def get_plex_with_update_coordinates(mesh):
    """
    Update the coordinates of the dmplex in mesh and return a copy of the dmplex
    """
    tdim = mesh.topological_dimension()
    gdim = mesh.geometric_dimension()
    entity_dofs = np.zeros(tdim + 1, dtype=np.int32)
    entity_dofs[0] = gdim
    coord_section, _ = mesh.create_section(entity_dofs)

    plex = mesh.topology_dm.clone()
    remove_pyop2_label(plex)

    coord_dm = plex.getCoordinateDM()
    coord_dm.setSection(coord_section)
    coords_local = coord_dm.createLocalVec()
    coords_local.array[:] = np.reshape(
        mesh.coordinates.dat.data_ro_with_halos, coords_local.array.shape
    )
    plex.setCoordinatesLocal(coords_local)

    return plex


def test_mesh_from_plex():
    mesh_init = RectangleMesh(8, 8, 1, 1)
    mesh_init.coordinates.dat.data[:] += 1

    plex = get_plex_with_update_coordinates(mesh_init)
    mesh = Mesh(plex)

    assert np.allclose(mesh.coordinates.dat.data, mesh_init.coordinates.dat.data)
