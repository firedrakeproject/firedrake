import numpy
from firedrake import *
from firedrake.mesh import plex_from_cell_list


def test_one_d_mesh_volume():
    mesh = IntervalMesh(2, 2)

    mesh.coordinates.dat.data[2] = 3
    extmesh = ExtrudedMesh(mesh, layers=[[0, 2], [2, 3]],
                           layer_height=1)

    assert numpy.allclose(assemble(1*dx(domain=extmesh)),
                          2 + 6)


def test_two_d_mesh_volume():
    dm = plex_from_cell_list(
        2,
        [[0, 1, 2],
         [1, 2, 3],
         [3, 4, 5],
         [1, 3, 6]],
        [[0, 0],
         [1, 0],
         [1, 1],
         [2, 0],
         [3, 0],
         [3, 1],
         [2, -1]],
        comm=COMM_WORLD
    )
    mesh2d = Mesh(dm, reorder=False)

    extmesh = ExtrudedMesh(mesh2d, [[0, 2],
                                    [1, 2],
                                    [3, 1],
                                    [2, 1]], layer_height=1)

    assert numpy.allclose(assemble(1*dx(domain=extmesh)),
                          0.5*(2 + 2 + 1 + 1))
