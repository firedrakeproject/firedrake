import pytest
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
