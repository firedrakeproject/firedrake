import numpy
from pyop2 import mpi
from firedrake import *
from firedrake.mesh import _from_cell_list as create_dm


def test_one_d_mesh_volume():
    mesh = IntervalMesh(2, 2)

    mesh.coordinates.dat.data[2] = 3
    extmesh = ExtrudedMesh(mesh, layers=[[0, 2], [2, 3]],
                           layer_height=1)

    assert numpy.allclose(assemble(1*dx(domain=extmesh)),
                          2 + 6)


def test_two_d_mesh_volume():
    with mpi.PyOP2Comm(COMM_WORLD) as comm:
        dm = create_dm(2, [[0, 1, 2],
                           [1, 2, 3],
                           [3, 4, 5],
                           [1, 3, 6]],
                       [[0, 0],
                        [1, 0],
                        [1, 1],
                        [2, 0],
                        [3, 0],
                        [3, 1],
                        [2, -1]], comm=comm)
    mesh2d = Mesh(dm, reorder=False)

    extmesh = ExtrudedMesh(mesh2d, [[0, 2],
                                    [1, 2],
                                    [3, 1],
                                    [2, 1]], layer_height=1)

    assert numpy.allclose(assemble(1*dx(domain=extmesh)),
                          0.5*(2 + 2 + 1 + 1))
