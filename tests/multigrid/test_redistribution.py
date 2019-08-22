import pytest
import numpy
from firedrake import *


@pytest.fixture(params=["none", "facet", "vertex"])
def source_overlap_type(request):
    if request.param == "none":
        return (DistributedMeshOverlapType.NONE, 0)
    elif request.param == "facet":
        return (DistributedMeshOverlapType.FACET, 1)
    elif request.param == "vertex":
        return (DistributedMeshOverlapType.VERTEX, 1)


@pytest.fixture(params=["none", "facet", "vertex"])
def target_overlap_type(request):
    if request.param == "none":
        return (DistributedMeshOverlapType.NONE, 0)
    elif request.param == "facet":
        return (DistributedMeshOverlapType.FACET, 1)
    elif request.param == "vertex":
        return (DistributedMeshOverlapType.VERTEX, 1)


@pytest.fixture(params=["single", "multiple"])
def source_mesh(request, source_overlap_type):
    if request.param == "single":
        comm = COMM_SELF
        make = COMM_WORLD.rank == 0
    elif request.param == "multiple":
        rank = COMM_WORLD.rank
        comm = COMM_WORLD.Split(rank % 2, rank)
        make = rank % 2 == 0

    if make:
        return UnitSquareMesh(5, 7, comm=comm, distribution_parameters={"overlap_type": source_overlap_type})
    else:
        return None


@pytest.fixture
def manager(source_mesh, target_overlap_type):
    return RedistributedMeshManager(source_mesh, COMM_WORLD, distribution_parameters={"overlap_type": target_overlap_type})


@pytest.fixture(params=[False, True], ids=["Scalar-FS", "VectorFS"])
def fstype(request):
    if request.param:
        return VectorFunctionSpace
    else:
        return FunctionSpace


@pytest.fixture
def source(source_mesh, fstype):
    if source_mesh is None:
        return None
    else:
        return Function(fstype(source_mesh, "P", 2))


@pytest.fixture
def target(manager, fstype):
    return Function(fstype(manager.target_mesh, "P", 2))


@pytest.fixture
def expr(fstype):
    if fstype == FunctionSpace:
        return lambda mesh: SpatialCoordinate(mesh)**2
    elif fstype == VectorFunctionSpace:
        return lambda mesh: as_vector([1 - SpatialCoordinate(mesh)[0], SpatialCoordinate(mesh)**2])


@pytest.mark.parallel(nprocs=3)
def test_source_to_target(source, target, manager, expr):
    if source is not None:
        source.interpolate(expr(source.ufl_domain()))

    manager.source_to_target(source, target)

    assert numpy.allclose(norm(expr(target.ufl_domain()) - target), 0)


@pytest.mark.parallel(nprocs=3)
def test_target_to_source(target, source, manager, expr):
    target.interpolate(expr(target.ufl_domain()))

    manager.target_to_source(target, source)

    if source is not None:
        result = numpy.allclose(norm(expr(source.ufl_domain()) - source), 0)
    else:
        result = None
    result = target.comm.bcast(result, root=manager.bcast_rank)
    assert result
