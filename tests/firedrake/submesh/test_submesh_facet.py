import pytest
import numpy as np
from firedrake import *
from firedrake.mesh import plex_from_cell_list
from pyop3.mpi import COMM_WORLD
from pytest_mpi.parallel_assert import parallel_assert


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


def _make_cube(overlap_type=None):
    # The simple partitioner makes the partition independent of the
    # installed partitioners.
    distribution_parameters = {"partitioner_type": "simple"}
    if overlap_type is not None:
        distribution_parameters["overlap_type"] = (overlap_type, 1)
    return UnitCubeMesh(4, 4, 4, distribution_parameters=distribution_parameters)


def _linear_function(subm):
    x, y, z = SpatialCoordinate(subm)
    V = FunctionSpace(subm, "DG", 1)
    f = Function(V)
    f.interpolate(1 + x + 2*y + 3*z)
    return f


def _make_codim1_submesh(overlap_type=None):
    # The y = 1 face of a unit cube.
    mesh = _make_cube(overlap_type)
    subm = Submesh(mesh, mesh.topological_dimension - 1, 4)
    return subm, _linear_function(subm)


def _make_codim2_submesh(overlap_type):
    # The boundary loop of the y = 1 face of a unit cube.
    face, _ = _make_codim1_submesh(overlap_type)
    loop = Submesh(face, subdomain_id="on_boundary")
    return loop, _linear_function(loop)


# Integrals of f = 1 + x + 2y + 3z on the y = 1 face, over the face (dx),
# its boundary (ds) and its interior edges (dS: grid lines and diagonals).
_codim1_dx = 5.
_codim1_ds = 20.
_codim1_dS = 30. + 20. * np.sqrt(2.)
# Sum of f over the 16 vertices of the boundary loop of the y = 1 face.
_codim2_dS = 80.


def _raises_overlap_error(f):
    try:
        assemble(f('-') * dS)
    except ValueError as e:
        return "overlap" in str(e)
    return False


@pytest.mark.parallel([2, 3])
def test_submesh_codim1_interior_facet_default_overlap_raises():
    # The default (FACET, 1) overlap is not enough for interior facet
    # integrals on a codim-1 submesh, so all ranks must raise.
    subm, f = _make_codim1_submesh()
    parallel_assert(_raises_overlap_error(f))


@pytest.mark.parallel([1, 2, 3])
@pytest.mark.parametrize("overlap_type", [DistributedMeshOverlapType.RIDGE,
                                          DistributedMeshOverlapType.VERTEX])
def test_submesh_codim1_interior_facet_ridge_vertex_overlap(overlap_type):
    subm, f = _make_codim1_submesh(overlap_type)
    assert np.isclose(assemble(f('+') * dS), _codim1_dS)
    assert np.isclose(assemble(f('-') * dS), _codim1_dS)


@pytest.mark.parallel([2, 3])
def test_submesh_codim1_interior_facet_relabeled_ridge_overlap():
    mesh = _make_cube(DistributedMeshOverlapType.RIDGE)
    V = FunctionSpace(mesh, "HDiv Trace", 0)
    marker = Function(V)
    bc = DirichletBC(V, 1, 4)
    bc.apply(marker)
    rmesh = RelabeledMesh(mesh, [marker], [111])
    subm = Submesh(rmesh, rmesh.topological_dimension - 1, 111)
    f = _linear_function(subm)
    assert np.isclose(assemble(f('-') * dS), _codim1_dS)


@pytest.mark.parallel([2, 3])
def test_submesh_codim2_interior_facet_ridge_overlap_raises():
    # A codim-2 submesh needs a VERTEX overlap of the parent mesh.
    loop, f = _make_codim2_submesh(DistributedMeshOverlapType.RIDGE)
    parallel_assert(_raises_overlap_error(f))


@pytest.mark.parallel([1, 2, 3])
def test_submesh_codim2_interior_facet_vertex_overlap():
    loop, f = _make_codim2_submesh(DistributedMeshOverlapType.VERTEX)
    assert np.isclose(assemble(f('+') * dS), _codim2_dS)
    assert np.isclose(assemble(f('-') * dS), _codim2_dS)


@pytest.mark.parallel([2, 3])
def test_submesh_ignore_halo_interior_facet_raises():
    # Without its halo, a submesh on the parent communicator misses the
    # second cell of the interior facets on the partition boundary.
    mesh = _make_cube()
    subm = Submesh(mesh, mesh.topological_dimension, ignore_halo=True)
    f = _linear_function(subm)
    parallel_assert(_raises_overlap_error(f))


@pytest.mark.parallel([1, 2])
def test_submesh_codim1_cell_exterior_facet_default_overlap():
    # Integrals that do not need interior facets still work with the default overlap.
    subm, f = _make_codim1_submesh()
    assert np.isclose(assemble(f * dx), _codim1_dx)
    assert np.isclose(assemble(f * ds), _codim1_ds)


def test_submesh_codim1_interior_facet_default_overlap_serial():
    subm, f = _make_codim1_submesh()
    assert np.isclose(assemble(f('-') * dS), _codim1_dS)
