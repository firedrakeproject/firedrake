from firedrake import *
import numpy as np


def remove_pyop2_label(plex):
    plex.removeLabel("firedrake_is_ghost")


def get_plex_with_update_coordinates(mesh):
    """
    Update the coordinates of the dmplex in mesh and return a copy of the dmplex
    """
    plex = mesh.topology_dm.clone()
    remove_pyop2_label(plex)

    coord_dm = plex.getCoordinateDM()
    coord_dm.setSection(mesh.coordinates.function_space().dm.getLocalSection())
    coords_local = coord_dm.createLocalVec()
    coords_local.array[...] = np.reshape(
        mesh.coordinates.dat.data_ro_with_halos, coords_local.array.shape
    )
    plex.setCoordinatesLocal(coords_local)

    return plex


def test_mesh_from_plex():
    mesh_init = RectangleMesh(8, 8, 1, 1)
    mesh_init.coordinates.dat.data_rw[...] += 1

    plex = get_plex_with_update_coordinates(mesh_init)
    mesh = Mesh(plex)

    assert np.allclose(mesh.coordinates.dat.data_ro, mesh_init.coordinates.dat.data_ro)
