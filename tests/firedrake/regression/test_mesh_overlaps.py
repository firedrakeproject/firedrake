from firedrake import *
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
    mesh.init()
    return mesh


@pytest.mark.parallel(nprocs=2)
def test_overlap(mesh, num_cells):
    assert mesh.num_cells() == num_cells


@pytest.mark.parallel(nprocs=2)
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

    mesh.init()
    fine_mesh.init()

    if overlap[0] == DistributedMeshOverlapType.NONE:
        assert mesh.num_cells() == 1
    else:
        assert mesh.num_cells() == 2

    assert fine_mesh.num_cells() == 4
