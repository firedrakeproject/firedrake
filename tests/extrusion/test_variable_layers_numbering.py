import pytest
import numpy
from firedrake import *
from firedrake.mesh import _from_cell_list as create_dm


def test_disconnected():
    mesh = UnitIntervalMesh(2)

    with pytest.raises(NotImplementedError):
        ExtrudedMesh(mesh, [[0, 1], [2, 1]],
                     layer_height=1)


def test_no_layers_property():
    mesh = UnitIntervalMesh(2)
    extmesh = ExtrudedMesh(mesh, [[0, 2], [1, 1]],
                           layer_height=1)
    with pytest.raises(ValueError):
        extmesh.layers


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
                       [0, 3, 8]).all()

    assert numpy.equal(bc_top.nodes,
                       [2, 7, 10]).all()


def test_numbering_one_d_P3():
    #           33--46-47-54
    #           |         |
    #           32  44 45 53
    #           31  42 43 52
    #           |         |
    #           30--40-41-51
    #           |         |
    #           29  38 39 50
    #           28  36 37 49
    #           |         |
    # 20--12-13-27--34-35-48
    # |         |
    # 19  10 11 26
    # 18  8  9  25
    # |         |
    # 17--6-7---24
    # |         |
    # 16  4 5   23
    # 15  2 3   22
    # |         |
    # 14--0-1---21
    mesh = UnitIntervalMesh(2)

    extmesh = ExtrudedMesh(mesh, layers=[[0, 2], [2, 2]],
                           layer_height=1)

    V = FunctionSpace(extmesh, "CG", 3)

    assert V.dof_dset.total_size == 55

    assert numpy.equal(V.cell_node_map().values,
                       [[21, 24, 22, 23, 14, 17, 15, 16,
                         0, 6, 2, 3, 1, 7, 4, 5],
                        [48, 51, 49, 50, 27, 30, 28, 29,
                         34, 40, 36, 37, 35, 41, 38, 39]]).all()

    assert numpy.equal(V.exterior_facet_node_map().values,
                       [[21, 24, 22, 23, 14, 17, 15, 16,
                         0, 6, 2, 3, 1, 7, 4, 5],
                        [48, 51, 49, 50, 27, 30, 28, 29,
                         34, 40, 36, 37, 35, 41, 38, 39]]).all()

    bc_left = DirichletBC(V, 0, 1)
    bc_right = DirichletBC(V, 0, 2)

    assert numpy.equal(bc_left.nodes,
                       [14, 15, 16, 17, 18, 19, 20]).all()

    assert numpy.equal(bc_right.nodes,
                       [48, 49, 50, 51, 52, 53, 54]).all()

    bc_bottom = DirichletBC(V, 0, "bottom")
    bc_top = DirichletBC(V, 0, "top")

    assert numpy.equal(bc_bottom.nodes,
                       [0, 1, 14, 21, 34, 35, 48]).all()

    assert numpy.equal(bc_top.nodes,
                       [12, 13, 20, 33, 46, 47, 54]).all()


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
    dm = create_dm(2, [[0, 1, 2],
                       [1, 2, 3],
                       [3, 4, 5]],
                   [[0, 0],
                    [1, 0],
                    [1, 1],
                    [2, 0],
                    [3, 0],
                    [3, 1]], COMM_WORLD)
    dm.markBoundaryFaces("boundary_ids")

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
                       [0, 3, 6, 9, 12, 14]).all()

    assert numpy.equal(bc_top.nodes,
                       [2, 5, 8, 11, 13, 15]).all()

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
    dm = create_dm(2, [[0, 1, 2],
                       [1, 2, 3],
                       [3, 4, 5]],
                   [[0, 0],
                    [1, 0],
                    [1, 1],
                    [2, 0],
                    [3, 0],
                    [3, 1]], COMM_WORLD)
    dm.markBoundaryFaces("boundary_ids")

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
                       [[12, 13, 15, 16, 18, 19,
                         6, 7, 9, 10, 3, 4, 0, 1],
                        [16, 17, 19, 20, 27, 28,
                         23, 24, 25, 26, 7, 8, 21, 22],
                        [28, 29, 38, 39, 40, 41,
                         34, 35, 36, 37, 32, 33, 30, 31]]).all()

    bc_bottom = DirichletBC(V, 0, "bottom")
    bc_top = DirichletBC(V, 0, "top")

    assert numpy.equal(bc_bottom.nodes,
                       [0, 3, 6, 9, 12, 15, 18, 21, 23, 25,
                        27, 30, 32, 34, 36, 38, 40]).all()

    assert numpy.equal(bc_top.nodes,
                       [2, 5, 8, 11, 14, 17, 20, 22, 24, 26,
                        29, 31, 33, 35, 37, 39, 41]).all()

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
                    [2, -1]], COMM_WORLD)
    dm.createLabel("boundary_ids")

    for faces, val in [((11, 13), 1),
                       ((14, 20), 2),
                       ((16, ), 3),
                       ((17, 18, 19), 4)]:
        for face in faces:
            dm.setLabelValue("boundary_ids", face, val)

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
                       [0, 3, 7, 11, 15, 17, 19]).all()

    assert numpy.equal(bc_top.nodes,
                       [2, 6, 10, 14, 16, 18, 20]).all()

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
