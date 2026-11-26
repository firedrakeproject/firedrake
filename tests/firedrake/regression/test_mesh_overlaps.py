from firedrake import *
from firedrake.mesh import PartitionOverlap
from pytest_mpi.parallel_assert import parallel_assert
import numpy as np
import pytest


@pytest.fixture(params=["NONE", "FACET", "VERTEX"])
def overlap_type(request):
    return getattr(DistributedMeshOverlapType, request.param)


@pytest.fixture
def overlap(overlap_type):
    if overlap_type == DistributedMeshOverlapType.NONE:
        return (overlap_type, 0)
    else:
        return (overlap_type, 1)


@pytest.fixture
def num_cells(overlap_type):
    if overlap_type == DistributedMeshOverlapType.NONE:
        return 4
    elif overlap_type == DistributedMeshOverlapType.FACET:
        return 6
    elif overlap_type == DistributedMeshOverlapType.VERTEX:
        return 7


@pytest.fixture(params=["flat", "refined"])
def mesh(request, overlap):
    if COMM_WORLD.rank == 0:
        if request.param == "refined":
            # Zero overlap distribution
            # +---+
            # |\ 1|
            # | \ |
            # |0 \|
            # +---+
            partition = ([1, 1], [0, 1])
        else:
            # Zero overlap distribution
            # +---+---+
            # |\ 1|\ 1|
            # | \ | \ |
            # |0 \|1 \|
            # +---+---+
            # |\ 0|\ 1|
            # | \ | \ |
            # |0 \|0 \|
            # +---+---+
            partition = ([4, 4], [0, 1, 2, 4, 3, 5, 6, 7])
    else:
        partition = (None, None)

    params = {"partition": partition,
              "overlap_type": overlap}
    if request.param == "refined":
        mesh = UnitSquareMesh(1, 1, reorder=False,
                              distribution_parameters=params)
        # Mesh hierarchy should inherit from mesh if no distribution
        # parameters are provided.
        mesh = MeshHierarchy(mesh, 1)[-1]
    else:
        mesh = UnitSquareMesh(2, 2, reorder=False,
                              distribution_parameters=params)
    return mesh


@pytest.mark.parallel(2)
def test_overlap(mesh, num_cells):
    assert mesh.num_cells() == num_cells


@pytest.mark.parallel(2)
def test_override_distribution_parameters(overlap):
    if COMM_WORLD.rank == 0:
        # Zero overlap distribution
        # +---+
        # |\ 1|
        # | \ |
        # |0 \|
        # +---+
        partition = ([1, 1], [0, 1])
    else:
        partition = (None, None)

    params = {"partition": partition,
              "overlap_type": overlap}
    mesh = UnitSquareMesh(1, 1, reorder=False,
                          distribution_parameters=params)
    # Mesh hierarchy should override distribution parameters from
    # coarse mesh if given.
    params = {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}
    fine_mesh = MeshHierarchy(mesh, 1, distribution_parameters=params)[-1]

    if overlap[0] == DistributedMeshOverlapType.NONE:
        assert mesh.num_cells() == 1
    else:
        assert mesh.num_cells() == 2

    assert fine_mesh.num_cells() == 4


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("reorder", [False, True])
def test_submesh_distribution_parameters(overlap, reorder):
    # Test that mesh._distribution_parameters and mesh._did_reordering
    # are propagated
    partition = True
    params = {"partition": partition,
              "overlap_type": overlap}
    mesh = UnitSquareMesh(2, 2, reorder=reorder,
                          distribution_parameters=params)
    orig_params = mesh._distribution_parameters
    did_reordering = mesh._did_reordering
    assert did_reordering == reorder

    x, *_ = SpatialCoordinate(mesh)
    DG0 = FunctionSpace(mesh, "DG", 0)
    ind = Function(DG0).interpolate(conditional(lt(x, 0.5), 1, 0))
    label = 111
    rmesh = RelabeledMesh(mesh, [ind], [label])
    assert rmesh._distribution_parameters == orig_params
    assert rmesh._did_reordering == did_reordering

    dim = mesh.topological_dimension
    submesh = Submesh(rmesh, dim, label)
    assert submesh._distribution_parameters == orig_params
    assert submesh._did_reordering == did_reordering


NONE = DistributedMeshOverlapType.NONE
FACET = DistributedMeshOverlapType.FACET
RIDGE = DistributedMeshOverlapType.RIDGE
VERTEX = DistributedMeshOverlapType.VERTEX


@pytest.mark.parametrize("overlap,codim,expected", [
    ((NONE, 0), 0, (NONE, 0)),
    ((NONE, 0), 1, (NONE, 0)),
    ((NONE, 0), 2, (NONE, 0)),
    ((FACET, 1), 0, (FACET, 1)),
    ((FACET, 1), 1, (NONE, 0)),
    ((FACET, 1), 2, (NONE, 0)),
    ((RIDGE, 1), 0, (RIDGE, 1)),
    ((RIDGE, 1), 1, (FACET, 1)),
    ((RIDGE, 1), 2, (NONE, 0)),
    ((VERTEX, 1), 0, (VERTEX, 1)),
    ((VERTEX, 1), 1, (VERTEX, 1)),
    ((VERTEX, 1), 2, (VERTEX, 1)),
])
def test_partition_overlap_for_codim(overlap, codim, expected):
    assert PartitionOverlap(*overlap).for_codim(codim) == PartitionOverlap(*expected)


@pytest.mark.parametrize("overlap,expected", [
    ((NONE, 0), False),
    ((FACET, 0), False),
    ((RIDGE, 0), False),
    ((FACET, 1), True),
    ((RIDGE, 1), True),
    ((VERTEX, 1), True),
    ((FACET, 2), True),
])
def test_partition_overlap_at_least_facet(overlap, expected):
    assert (PartitionOverlap(FACET, 1) <= PartitionOverlap(*overlap)) == expected


@pytest.mark.parallel([2, 3])
def test_interior_facet_none_overlap_raises():
    mesh = UnitSquareMesh(4, 4, distribution_parameters={"overlap_type": (NONE, 0)})
    try:
        assemble(Constant(1) * dS(domain=mesh))
    except ValueError as e:
        raised = "overlap" in str(e)
    else:
        raised = False
    parallel_assert(raised)


@pytest.mark.parallel([1, 2])
def test_relabeled_extruded_interior_facet_vert():
    mesh = UnitSquareMesh(4, 4)
    M = FunctionSpace(mesh, "DG", 0)
    marker = Function(M)
    marker.assign(1)
    mesh = RelabeledMesh(mesh, [marker], [111])
    mesh = ExtrudedMesh(mesh, 2, layer_height=0.5)
    # Interior edges of UnitSquareMesh(4, 4), 3 + 3 grid lines and 16 diagonals, times the height 1.
    assert np.isclose(assemble(Constant(1) * dS_v(domain=mesh)), 6 + 4 * np.sqrt(2))
