import pytest
import numpy
from firedrake import *
from firedrake.__future__ import *
from firedrake.mesh import plex_from_cell_list
from firedrake.utils import IntType


def test_disconnected():
    mesh = UnitIntervalMesh(2)

    with pytest.raises(NotImplementedError):
        ExtrudedMesh(mesh, [[0, 1], [2, 1]],
                     layer_height=1)


def test_no_layer_height():
    mesh = UnitIntervalMesh(2)

    with pytest.raises(ValueError):
        ExtrudedMesh(mesh, [[0, 2], [1, 1]])


def test_mismatch_layers_array():
    mesh = UnitIntervalMesh(3)

    with pytest.raises(ValueError):
        ExtrudedMesh(mesh, [[0, 2], [1, 1]])


def test_numbering_one_d_P1():
    #      7----10
    #      |    |
    #      |    |
    #      6----9
    #      |    |
    #      |    |
    # 2----5----8
    # |    |
    # |    |
    # 1----4
    # |    |
    # |    |
    # 0----3
    mesh = UnitIntervalMesh(2)

    extmesh = ExtrudedMesh(mesh, layers=[[0, 2], [2, 2]],
                           layer_height=1)

    V = FunctionSpace(extmesh, "CG", 1)

    assert V.dof_dset.total_size == 11

    assert numpy.equal(V.cell_node_map().values,
                       [[3, 4, 0, 1],
                        [8, 9, 5, 6]]).all()

    assert numpy.equal(V.exterior_facet_node_map().values,
                       [[3, 4, 0, 1],
                        [8, 9, 5, 6]]).all()

    bc_left = DirichletBC(V, 0, 1)
    bc_right = DirichletBC(V, 0, 2)

    assert numpy.equal(bc_left.nodes,
                       [0, 1, 2]).all()

    assert numpy.equal(bc_right.nodes,
                       [8, 9, 10]).all()

    bc_bottom = DirichletBC(V, 0, "bottom")
    bc_top = DirichletBC(V, 0, "top")

    assert numpy.equal(bc_bottom.nodes,
                       [0, 3, 4, 5, 8]).all()

    assert numpy.equal(bc_top.nodes,
                       [2, 5, 6, 7, 10]).all()


def test_numbering_one_d_P3():
    #           33--46-47-54
    #           |         |
    #           32  43 45 53
    #           31  42 44 52
    #           |         |
    #           30--40-41-51
    #           |         |
    #           29  37 39 50
    #           28  36 38 49
    #           |         |
    # 20--12-13-27--34-35-48
    # |         |
    # 19  9 11  26
    # 18  8 10  25
    # |         |
    # 17--6-7---24
    # |         |
    # 16  3 5   23
    # 15  2 4   22
    # |         |
    # 14--0-1---21
    mesh = UnitIntervalMesh(2)

    extmesh = ExtrudedMesh(mesh, layers=[[0, 2], [2, 2]],
                           layer_height=1)

    fe = FiniteElement("CG", extmesh.ufl_cell(), 3, variant="equispaced")
    V = FunctionSpace(extmesh, fe)

    assert V.dof_dset.total_size == 55

    assert numpy.equal(V.cell_node_map().values,
                       [[21, 24, 22, 23, 14, 17, 15, 16,
                         1, 7, 4, 5, 0, 6, 2, 3],
                        [48, 51, 49, 50, 27, 30, 28, 29,
                         35, 41, 38, 39, 34, 40, 36, 37]]).all()

    assert numpy.equal(V.exterior_facet_node_map().values,
                       [[21, 24, 22, 23, 14, 17, 15, 16,
                         1, 7, 4, 5, 0, 6, 2, 3],
                        [48, 51, 49, 50, 27, 30, 28, 29,
                         35, 41, 38, 39, 34, 40, 36, 37]]).all()

    bc_left = DirichletBC(V, 0, 1)
    bc_right = DirichletBC(V, 0, 2)

    assert numpy.equal(bc_left.nodes,
                       [14, 15, 16, 17, 18, 19, 20]).all()

    assert numpy.equal(bc_right.nodes,
                       [48, 49, 50, 51, 52, 53, 54]).all()

    bc_bottom = DirichletBC(V, 0, "bottom")
    bc_top = DirichletBC(V, 0, "top")

    assert numpy.equal(bc_bottom.nodes,
                       [0, 1, 14, 21, 22, 23, 24, 25, 26, 27, 34, 35, 48]).all()

    assert numpy.equal(bc_top.nodes,
                       [12, 13, 20, 27, 28, 29, 30, 31, 32, 33, 46, 47, 54]).all()


def test_numbering_two_d_P1():
    #
    #   Top view          Side view
    #                           x---x
    #                           |   |
    #     2       5     x---x---x---x
    #    /|\     /|     |   |   |
    #   / | \   / |     x---x---x
    #  /  |  \ /  |     |   |
    # 0---1---3---4     x---x
    dm = plex_from_cell_list(
        2,
        [[0, 1, 2],
         [1, 2, 3],
         [3, 4, 5]],
        [[0, 0],
         [1, 0],
         [1, 1],
         [2, 0],
         [3, 0],
         [3, 1]],
        comm=COMM_WORLD
    )
    dm.markBoundaryFaces("Face Sets")

    mesh2d = Mesh(dm, reorder=False)

    extmesh = ExtrudedMesh(mesh2d, [[0, 2],
                                    [1, 1],
                                    [2, 1]],
                           layer_height=1)

    V = FunctionSpace(extmesh, "CG", 1)

    assert V.dof_dset.size == 16
    assert numpy.equal(V.cell_node_map().values,
                       [[0, 1, 3, 4, 6, 7],
                        [4, 5, 7, 8, 9, 10],
                        [10, 11, 12, 13, 14, 15]]).all()

    bc_bottom = DirichletBC(V, 0, "bottom")
    bc_top = DirichletBC(V, 0, "top")

    assert numpy.equal(bc_bottom.nodes,
                       [0, 3, 4, 6, 7, 9, 10, 12, 14]).all()

    assert numpy.equal(bc_top.nodes,
                       [2, 5, 8, 10, 11, 13, 15]).all()

    bc_side = DirichletBC(V, 0, 1)

    assert numpy.equal(bc_side.nodes,
                       numpy.arange(16)).all()


def test_numbering_two_d_P2BxP1():
    #
    #   Top view          Side view
    #                           x---x
    #                           |   |
    #     2       5     x---x---x---x
    #    /|\     /|     |   |   |
    #   / | \   / |     x---x---x
    #  /  |  \ /  |     |   |
    # 0---1---3---4     x---x
    dm = plex_from_cell_list(
        2,
        [[0, 1, 2],
         [1, 2, 3],
         [3, 4, 5]],
        [[0, 0],
         [1, 0],
         [1, 1],
         [2, 0],
         [3, 0],
         [3, 1]],
        comm=COMM_WORLD
    )
    dm.markBoundaryFaces("Face Sets")

    mesh2d = Mesh(dm, reorder=False)

    extmesh = ExtrudedMesh(mesh2d, [[0, 2],
                                    [1, 1],
                                    [2, 1]],
                           layer_height=1)

    U = FiniteElement("CG", triangle, 2)
    B = FiniteElement("B", triangle, 3)
    V = FiniteElement("CG", interval, 1)
    W = TensorProductElement(U+B, V)
    V = FunctionSpace(extmesh, W)

    assert V.dof_dset.size == 42
    assert numpy.equal(V.cell_node_map().values,
                       [[12, 13, 3, 4, 9, 10, 15, 16, 6, 7, 18, 19, 0, 1],
                        [16, 17, 7, 8, 25, 26, 19, 20, 23, 24, 27, 28, 21, 22],
                        [28, 29, 32, 33, 36, 37, 38, 39, 34, 35, 40, 41, 30, 31]]).all()

    bc_bottom = DirichletBC(V, 0, "bottom")
    bc_top = DirichletBC(V, 0, "top")

    assert numpy.equal(bc_bottom.nodes,
                       [0, 3, 6, 7, 9, 12, 15, 16, 18, 19,
                        21, 23, 25, 27, 28, 30, 32, 34, 36,
                        38, 40]).all()

    assert numpy.equal(bc_top.nodes,
                       [2, 5, 8, 11, 14, 17, 20, 22, 24, 26,
                        28, 29, 31, 33, 35, 37, 39, 41]).all()

    bc_side = DirichletBC(V, 0, 1)

    assert numpy.equal(bc_side.nodes,
                       [3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41]).all()


def test_numbering_two_d_bigger():
    #
    #    Top view, plex points
    #       6           9
    #      /|\         /|
    #     / | \       / |
    #    13 |  14    18 |
    #   /   12  \   /   17
    #  / 0  | 1  \ /  2 |
    # 4--11-5--15-7--16-8
    #        \  3 |
    #         \   19
    #          20 |
    #           \ |
    #            \|
    #             10
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
    dm.createLabel("Face Sets")

    for faces, val in [((11, 13), 1),
                       ((14, 20), 2),
                       ((16, ), 3),
                       ((17, 18, 19), 4),
                       # This one is an interior face
                       ((12, ), 5)]:
        for face in faces:
            dm.setLabelValue("Face Sets", face, val)

    mesh2d = Mesh(dm, reorder=False)

    extmesh = ExtrudedMesh(mesh2d, [[0, 2],
                                    [1, 2],
                                    [3, 1],
                                    [2, 1]], layer_height=1)

    V = FunctionSpace(extmesh, "CG", 1)

    assert V.dof_dset.size == 21
    assert numpy.equal(V.cell_node_map().values,
                       [[0, 1, 3, 4, 7, 8],
                        [4, 5, 8, 9, 11, 12],
                        [13, 14, 15, 16, 17, 18],
                        [5, 6, 12, 13, 19, 20]]).all()

    bc_bottom = DirichletBC(V, 0, "bottom")
    bc_top = DirichletBC(V, 0, "top")

    assert numpy.equal(bc_bottom.nodes,
                       [0, 3, 4, 5, 7, 8, 11, 12, 13, 15, 17, 19]).all()

    assert numpy.equal(bc_top.nodes,
                       [2, 5, 6, 9, 10, 13, 14, 16, 18, 20]).all()

    bc_side = DirichletBC(V, 0, "on_boundary")

    assert numpy.equal(bc_side.nodes,
                       numpy.arange(21)).all()

    assert numpy.equal(DirichletBC(V, 0, 1).nodes,
                       [0, 1, 2, 3, 4, 5, 7, 8, 9]).all()

    assert numpy.equal(DirichletBC(V, 0, 2).nodes,
                       [5, 6, 8, 9, 10, 11, 12, 13, 19, 20]).all()

    assert numpy.equal(DirichletBC(V, 0, 3).nodes,
                       [13, 14, 15, 16]).all()

    assert numpy.equal(DirichletBC(V, 0, 3).nodes,
                       [13, 14, 15, 16]).all()

    assert numpy.equal(DirichletBC(V, 0, 4).nodes,
                       [12, 13, 14, 15, 16, 17, 18, 19, 20]).all()

    # Interior face between base plex cells 0 and 1.
    assert numpy.equal(DirichletBC(V, 0, 5).nodes,
                       [4, 5, 8, 9]).all()


def test_numbering_quad():
    # Number of cells in each column.
    #               Side 4
    #         +-------+-------+
    #         |       |       |
    #         |   1   |   2   |
    #         |       |       |
    # Side 1  +-------+-------+  Side 2
    #         |       |       |
    #         |   2   |   1   |
    #         |       |       |
    #         +-------+-------+
    #               Side 3
    mesh = UnitSquareMesh(2, 2, quadrilateral=True)
    extmesh = ExtrudedMesh(mesh, layers=[[0, 2], [0, 1], [0, 1], [0, 2]],
                           layer_height=1)
    V = FunctionSpace(extmesh, "Q", 1)
    assert numpy.equal(V.cell_node_map().values,
                       [[0, 1, 3, 4, 9, 10, 6, 7],
                        [9, 10, 6, 7, 15, 16, 12, 13],
                        [3, 4, 17, 18, 6, 7, 19, 20],
                        [6, 7, 19, 20, 12, 13, 22, 23]]).all()

    assert numpy.equal(DirichletBC(V, 0, "bottom").nodes,
                       [0, 3, 6, 9, 12, 15, 17, 19, 22]).all()

    assert numpy.equal(DirichletBC(V, 0, "top").nodes,
                       [2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 18, 20, 21, 24]).all()

    assert numpy.equal(DirichletBC(V, 0, 1).nodes,
                       [0, 1, 2, 3, 4, 5, 17, 18]).all()

    assert numpy.equal(DirichletBC(V, 0, 2).nodes,
                       [12, 13, 14, 15, 16, 22, 23, 24]).all()

    assert numpy.equal(DirichletBC(V, 0, 3).nodes,
                       [0, 1, 2, 9, 10, 11, 15, 16]).all()

    assert numpy.equal(DirichletBC(V, 0, 4).nodes,
                       [17, 18, 19, 20, 21, 22, 23, 24]).all()


@pytest.mark.parametrize(["domain", "expected"],
                         [("top", [3, 6, 7, 9, 11, 13, 14, 17]),
                          ("bottom", [0, 4, 5, 8, 10, 12, 15]),
                          (1, [0, 1, 2, 3]),
                          (2, [15, 16, 17])])
def test_bcs_nodes(domain, expected):
    # 3----7              14---17
    # |    |              |    |
    # |    |              |    |
    # 2----6----9----11---13---16
    # |    |    |    |    |    |
    # |    |    |    |    |    |
    # 1----5----8----10---12---15
    # |    |
    # |    |
    # 0----4
    mesh = UnitIntervalMesh(5)
    V = VectorFunctionSpace(mesh, "DG", 0, dim=2)

    x, = SpatialCoordinate(mesh)

    selector = assemble(interpolate(
        conditional(
            real(x) < 0.2,
            as_vector([0, 3]),
            conditional(real(x) > 0.8,
                        as_vector([1, 2]),
                        as_vector([1, 1]))),
        V))

    layers = numpy.empty((5, 2), dtype=IntType)

    layers[:] = selector.dat.data_ro.real

    extmesh = ExtrudedMesh(mesh, layers=layers,
                           layer_height=0.25)

    V = FunctionSpace(extmesh, "CG", 1)

    nodes = DirichletBC(V, 0, domain).nodes

    assert numpy.equal(nodes, expected).all()


@pytest.mark.parallel(nprocs=4)
def test_layer_extents_parallel():
    # +-----+-----+
    # |\  1 |\  3 |  cell_layers = [[0, 1],
    # | \   | \   |                 [0, 1],
    # |  \  |  \  |                 [0, 1],
    # |   \ |   \ |                 [0, 2]]
    # | 0  \| 2  \|
    # +-----+-----+
    #
    # Cell ownership (rank -> cell):
    # 0 -> 1
    # 1 -> 0
    # 2 -> 3
    # 3 -> 2
    if COMM_WORLD.rank == 0:
        sizes = numpy.asarray([1, 1, 1, 1], dtype=IntType)
        points = numpy.asarray([1, 0, 3, 2], dtype=IntType)
    else:
        sizes = None
        points = None

    mesh = UnitSquareMesh(2, 1, reorder=False, distribution_parameters={"partition":
                                                                        (sizes, points)})
    V = FunctionSpace(mesh, "DG", 0)
    x, _ = SpatialCoordinate(mesh)
    selector = assemble(interpolate(x - 0.5, V))
    layers = numpy.empty((mesh.num_cells(), 2), dtype=IntType)
    data = selector.dat.data_ro_with_halos.real
    for cell in V.cell_node_map().values_with_halo:
        if data[cell] < 0.25:
            layers[cell, :] = [0, 1]
        else:
            layers[cell, :] = [0, 2]
    extmesh = ExtrudedMesh(mesh, layers=layers, layer_height=1)
    if mesh.comm.rank == 0:
        #  Top view, plex points
        #  4--8--6
        #  |\  0 |\
        #  | \   | \
        #  9  10 12 13
        #  |   \ |   \
        #  | 1  \| 2  \
        #  3--11-5--14-7
        expected = numpy.asarray([
            # cells
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            # vertices
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 3, 0, 2],
            [0, 3, 0, 2],
            # edges
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 3, 0, 2],
            [0, 2, 0, 2]], dtype=IntType)
    elif mesh.comm.rank == 1:
        #  Top view, plex points
        #  3--9--5
        #  |\  1 |
        #  | \   |
        #  6  7  10
        #  |   \ |
        #  | 0  \|
        #  2--8--4
        expected = numpy.asarray([
            # cells
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            # vertices
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 3, 0, 2],
            # edges
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2]], dtype=IntType)
    elif mesh.comm.rank == 2:
        #  Top view, plex points
        #  4--6--2
        #  |\  0 |
        #  | \   |
        #  9  8  7
        #  |   \ |
        #  | 1  \|
        #  3--10-5
        expected = numpy.asarray([
            # cells
            [0, 3, 0, 3],
            [0, 2, 0, 2],
            # vertices
            [0, 3, 0, 3],
            [0, 2, 0, 2],
            [0, 3, 0, 2],
            [0, 3, 0, 2],
            # edges
            [0, 3, 0, 3],
            [0, 3, 0, 3],
            [0, 2, 0, 2],
            [0, 3, 0, 2],
            [0, 2, 0, 2]], dtype=IntType)
    elif mesh.comm.rank == 3:
        #  Top view, plex points
        #  6--11-4--13-7
        #   \  1 |\  2 |
        #    \   | \   |
        #    12  8  9 14
        #      \ |   \ |
        #       \| 0  \|
        #        3--10-5
        expected = numpy.asarray([
            # cells
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 3, 0, 3],
            # vertices
            [0, 2, 0, 2],
            [0, 3, 0, 2],
            [0, 3, 0, 2],
            [0, 2, 0, 2],
            [0, 3, 0, 3],
            # edges
            [0, 2, 0, 2],
            [0, 3, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 3, 0, 3],
            [0, 3, 0, 3]], dtype=IntType)
    assert numpy.equal(extmesh.layer_extents, expected).all()

    V = FunctionSpace(extmesh, "CG", 1)

    assert V.dof_dset.layout_vec.getSize() == 15


@pytest.mark.parallel(nprocs=3)
def test_layer_extents_parallel_vertex_owners():
    dm = plex_from_cell_list(
        2,
        [[0, 1, 2],
         [1, 2, 3],
         [2, 3, 4]],
        [[0, 0],
         [1, 0],
         [0, 1],
         [1, 1],
         [2, 0]],
        comm=COMM_WORLD
    )

    if COMM_WORLD.rank == 0:
        sizes = numpy.asarray([1, 1, 1], dtype=IntType)
        points = numpy.asarray([0, 1, 2], dtype=IntType)
    else:
        sizes = None
        points = None

    mesh = Mesh(dm, reorder=False, distribution_parameters={"partition":
                                                            (sizes, points)})
    V = FunctionSpace(mesh, "DG", 0)

    x, _ = SpatialCoordinate(mesh)
    selector = assemble(interpolate(x, V))

    layers = numpy.empty((mesh.num_cells(), 2), dtype=IntType)

    data = selector.dat.data_ro_with_halos.real
    for cell in V.cell_node_map().values_with_halo:
        if data[cell] < 0.5:
            layers[cell, :] = [1, 1]
        else:
            layers[cell, :] = [0, 3]

    extmesh = ExtrudedMesh(mesh, layers=layers, layer_height=1)

    if mesh.comm.rank == 0:
        #  Top view, plex points
        #  3--9--5
        #  |\  1 |
        #  | \   |
        #  7  8 10
        #  |   \ |
        #  | 0  \|
        #  2--6--4
        expected = numpy.asarray([
            # cells
            [1, 3, 1, 3],
            [0, 4, 0, 4],
            # vertices
            [1, 3, 1, 3],
            [0, 4, 1, 3],
            [0, 4, 1, 3],
            [0, 4, 0, 4],
            # edges
            [1, 3, 1, 3],
            [1, 3, 1, 3],
            [0, 4, 1, 3],
            [0, 4, 0, 4],
            [0, 4, 0, 4]], dtype=IntType)
    elif mesh.comm.rank == 1:
        #  Top view, plex points
        #  3--9--6
        #  |\  0 |\
        #  | \   | \
        #  11 8 12  13
        #  |   \ |   \
        #  | 1  \| 2  \
        #  4--10-5--14-7
        expected = numpy.asarray([
            # cells
            [0, 4, 0, 4],
            [1, 3, 1, 3],
            [0, 4, 0, 4],
            # vertices
            [0, 4, 1, 3],
            [1, 3, 1, 3],
            [0, 4, 1, 3],
            [0, 4, 0, 4],
            [0, 4, 0, 4],
            # edges
            [0, 4, 1, 3],
            [0, 4, 0, 4],
            [1, 3, 1, 3],
            [1, 3, 1, 3],
            [0, 4, 0, 4],
            [0, 4, 0, 4],
            [0, 4, 0, 4]], dtype=IntType)
    elif mesh.comm.rank == 2:
        #  Top view, plex points
        #  5--10-3
        #   \  1 |\
        #    \   | \
        #     9  6  7
        #      \ |   \
        #       \| 0  \
        #        2--8--4
        expected = numpy.asarray([
            # cells
            [0, 4, 0, 4],
            [0, 4, 0, 4],
            # vertices
            [0, 4, 1, 3],
            [0, 4, 0, 4],
            [0, 4, 0, 4],
            [0, 4, 1, 3],
            # edges
            [0, 4, 0, 4],
            [0, 4, 0, 4],
            [0, 4, 0, 4],
            [0, 4, 1, 3],
            [0, 4, 0, 4]], dtype=IntType)

    assert numpy.equal(extmesh.layer_extents, expected).all()

    V = FunctionSpace(extmesh, "CG", 1)

    assert V.dof_dset.layout_vec.getSize() == 18
