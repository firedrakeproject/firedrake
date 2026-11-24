import pytest
import numpy as np
from firedrake import *
from firedrake.mesh import plex_from_cell_list
from pyop3.mpi import COMM_WORLD


@pytest.mark.parallel(2)
def test_submesh_facet_corner_case_1():
    #  mesh and ownership:
    #
    #      +------+
    #     /   0  /|
    #    +------+------+
    #   /   0  /  1   /|
    #  +------+------+ |
    #  |      |      | +
    #  |      |      |/
    #  +------+------+
    #
    #  If (DistributedMeshOverlapType.FACET, 1) was used for the parent mesh:
    #
    #  subm would look like:
    #
    #             +
    #            /|
    #           +------+        +------+
    #           | +    |        |      |
    #           |/     |        |      |
    #           +------+        +------+
    #
    #            rank 0          rank 1
    #
    #  This is sane, but current quad orientation implementation
    #  can not handle such asymmetric overlaps.
    #
    #  If (DistributedMeshOverlapType.RIDGE, 1) is used for the parent mesh:
    #
    #  subm looks like:
    #
    #             +               +
    #            /|              /|
    #           +------+        +------+
    #           | +    |        | +    |
    #           |/     |        |/     |
    #           +------+        +------+
    #
    #            rank 0          rank 1
    #
    #  This works, and this is better.
    vertices = [
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 1.],
        [0., 2., 0.],
        [0., 2., 1.],
        [1., 0., 0.],
        [1., 0., 1.],
        [1., 1., 0.],
        [1., 1., 1.],
        [1., 2., 0.],
        [1., 2., 1.],
        [2., 0., 0.],
        [2., 0., 1.],
        [2., 1., 0.],
        [2., 1., 1.],
    ]
    cell_vert_map = [
        [0, 2, 8, 6, 1, 7, 9, 3],
        [2, 4, 10, 8, 3, 9, 11, 5],
        [6, 8, 14, 12, 7, 13, 15, 9],
    ]
    plex = plex_from_cell_list(3, cell_vert_map, vertices, COMM_WORLD)
    mesh = Mesh(
        plex,
        distribution_parameters={
            "partition": True,
            "partitioner_type": "simple",
            "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
        },
    )
    x, y, z = SpatialCoordinate(mesh)
    facet0 = And(And(x > 0.9, x < 1.1), y > 0.9)
    facet1 = And(And(y > 0.9, y < 1.1), x > 0.9)
    V = FunctionSpace(mesh, "Q", 2)
    f = Function(V).interpolate(conditional(Or(facet0, facet1), 1, 0))
    mesh = RelabeledMesh(mesh, [f], [999])
    subm = Submesh(mesh, mesh.topological_dimension - 1, 999)
    v = assemble(Constant(1.) * dx(domain=subm))
    assert abs(v - 2.) < 2.e-15


@pytest.mark.parallel(nprocs=2)
def test_submesh_facet_corner_case_2():
    #  Naively, one would have a submesh like:
    #
    #  +--------+    +--------+
    #  |       /|            /|
    #  |    /   |         /   |
    #  | /      |      /      |
    #  +--------+    +--------+
    #    rank 0        rank 1
    #
    #  , which might be fine in general, but
    #  Firedrake does not expect this kind of mesh.
    #  Thus, in dmcommon.create_submesh(), we cull
    #  redundant lower-dimensional points in the
    #  label to obtain a submesh like:
    #
    #  +--------+             +
    #  |       /|            /|
    #  |    /   |         /   |
    #  | /      |      /      |
    #  +--------+    +--------+
    #    rank 0        rank 1
    mesh = UnitCubeMesh(1, 1, 1)
    V = FunctionSpace(mesh, "HDiv Trace", 0)
    x, y, z = SpatialCoordinate(mesh)
    facet_function = Function(V).interpolate(
        conditional(x > .999, 1., 0.,)
    )
    facet_value = 999
    mesh = RelabeledMesh(mesh, [facet_function], [facet_value])
    _ = Submesh(mesh, mesh.topological_dimension - 1, facet_value)


def test_submesh_facet_all_facets():
    mesh = UnitCubeMesh(2, 2, 2)
    submesh1 = Submesh(mesh, mesh.topological_dimension - 1)

    V = FunctionSpace(mesh, "HDiv Trace", 0)
    facet_function = Function(V).assign(1)
    facet_value = 999
    rmesh = RelabeledMesh(mesh, [facet_function], [facet_value])
    submesh2 = Submesh(rmesh, mesh.topological_dimension - 1, facet_value)
    assert submesh2.cells.owned.local_size == submesh1.cells.owned.local_size


def _make_codim1_submesh(overlap_type=None):
    # The y = 1 face of a unit cube. The simple partitioner makes the
    # partition, and so the facets with a missing cell, independent of the
    # installed partitioners.
    distribution_parameters = {"partitioner_type": "simple"}
    if overlap_type is not None:
        distribution_parameters["overlap_type"] = (overlap_type, 1)
    mesh = UnitCubeMesh(4, 4, 4, distribution_parameters=distribution_parameters)
    subm = Submesh(mesh, mesh.topological_dimension - 1, 4)
    x, y, z = SpatialCoordinate(subm)
    f = Function(FunctionSpace(subm, "DG", 1)).interpolate(1 + x + 2*y + 3*z)
    return subm, f


# Integrals of f = 1 + x + 2y + 3z on the y = 1 face, over the face (dx),
# its boundary (ds) and its interior edges (dS: grid lines and diagonals).
_codim1_dx = 5.
_codim1_ds = 20.
_codim1_dS = 30. + 20. * np.sqrt(2.)


@pytest.mark.parallel(2)
def test_submesh_codim1_interior_facet_default_overlap_raises():
    # With the default (FACET, 1) overlap some owned interior facets of the
    # submesh have only one local cell. All ranks must raise.
    subm, f = _make_codim1_submesh()
    try:
        assemble(f('-') * dS)
    except ValueError as e:
        raised = "overlap" in str(e)
    else:
        raised = False
    assert all(subm.comm.allgather(raised))


@pytest.mark.parallel(2)
def test_submesh_codim0_interior_facet_no_overlap_raises():
    # A codim-0 submesh of a parent without overlap has the same problem.
    mesh = UnitSquareMesh(8, 8, distribution_parameters={
        "partitioner_type": "simple",
        "overlap_type": (DistributedMeshOverlapType.NONE, 0),
    })
    x, y = SpatialCoordinate(mesh)
    marker = Function(FunctionSpace(mesh, "DG", 0)).interpolate(conditional(lt(x, 0.5), 1, 0))
    mesh.mark_entities(marker, 111)
    subm = Submesh(mesh, mesh.topological_dimension, 111)
    x, y = SpatialCoordinate(subm)
    f = Function(FunctionSpace(subm, "DG", 1)).interpolate(1 + x + 2*y)
    try:
        assemble(f('-') * dS)
    except ValueError as e:
        raised = "overlap" in str(e)
    else:
        raised = False
    assert all(subm.comm.allgather(raised))


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("overlap_type", [DistributedMeshOverlapType.RIDGE,
                                          DistributedMeshOverlapType.VERTEX])
def test_submesh_codim1_interior_facet_ridge_vertex_overlap(overlap_type):
    subm, f = _make_codim1_submesh(overlap_type)
    assert np.isclose(assemble(f('+') * dS), _codim1_dS)
    assert np.isclose(assemble(f('-') * dS), _codim1_dS)


@pytest.mark.parallel([1, 2])
def test_submesh_codim1_cell_exterior_facet_default_overlap():
    # Integrals that do not need interior facets still work with the default overlap.
    subm, f = _make_codim1_submesh()
    assert np.isclose(assemble(f * dx), _codim1_dx)
    assert np.isclose(assemble(f * ds), _codim1_ds)


def test_submesh_codim1_interior_facet_default_overlap_serial():
    subm, f = _make_codim1_submesh()
    assert np.isclose(assemble(f('-') * dS), _codim1_dS)
