from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI


# Utility Functions

def cell_midpoints(m):
    """Get the coordinates of the midpoints of every cell in mesh `m`.

    :param m: The mesh to generate cell midpoints for.

    :returns: A tuple of numpy arrays `(midpoints, local_midpoints)` where
    `midpoints` are the midpoints for the entire mesh even if the mesh is
    distributed and `local_midpoints` are the midpoints of only the
    rank-local non-ghost cells."""
    if isinstance(m.topology, mesh.ExtrudedMeshTopology):
        raise NotImplementedError("Extruded meshes are not supported")
    m.init()
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(SpatialCoordinate(m))
    # since mesh may be distributed, the number of cells on the MPI rank
    # may not be the same on all ranks (note we exclude ghost cells
    # hence using num_cells_local = m.cell_set.size). Below local means
    # MPI rank local.
    num_cells_local = m.cell_set.size
    num_cells = MPI.COMM_WORLD.allreduce(num_cells_local, op=MPI.SUM)
    local_midpoints = f.dat.data_ro
    local_midpoints_size = np.array(local_midpoints.size)
    local_midpoints_sizes = np.empty(MPI.COMM_WORLD.size, dtype=int)
    MPI.COMM_WORLD.Allgatherv(local_midpoints_size, local_midpoints_sizes)
    midpoints = np.empty((num_cells, m.ufl_cell().geometric_dimension()), dtype=float)
    MPI.COMM_WORLD.Allgatherv(local_midpoints, (midpoints, local_midpoints_sizes))
    assert len(np.unique(midpoints, axis=0)) == len(midpoints)
    return midpoints, local_midpoints


@pytest.fixture(params=[pytest.param("interval", marks=pytest.mark.xfail(reason="swarm not implemented in 1d")),
                        "square",
                        pytest.param("extruded", marks=pytest.mark.xfail(reason="extruded meshes not supported")),
                        "cube",
                        "tetrahedron",
                        pytest.param("immersedsphere", marks=pytest.mark.skip(reason="immersed parent meshes not supported and will segfault PETSc when creating the DMSwarm")),
                        pytest.param("periodicrectangle", marks=pytest.mark.skip(reason="periodic meshes do not work properly with swarm creation"))])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(1, 1), 1)
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        return UnitIcosahedralSphereMesh()
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)


# pic swarm tests

def test_pic_swarm_in_plex(parentmesh):
    """Generate points in cell midpoints of mesh `parentmesh` and check correct
    swarm is created in plex."""

    # Setup

    parentmesh.init()
    inputpointcoords, inputlocalpointcoords = cell_midpoints(parentmesh)
    plex = parentmesh.topology._topology_dm
    from firedrake.petsc import PETSc
    fields = [("fieldA", 1, PETSc.IntType), ("fieldB", 2, PETSc.ScalarType)]
    swarm = mesh._pic_swarm_in_plex(plex, inputpointcoords, fields=fields)
    # Get point coords on current MPI rank
    localpointcoords = np.copy(swarm.getField("DMSwarmPIC_coor"))
    swarm.restoreField("DMSwarmPIC_coor")
    if len(inputpointcoords.shape) > 1:
        localpointcoords = np.reshape(localpointcoords, (-1, inputpointcoords.shape[1]))
    # Turn this into a number of points locally and MPI globally before
    # doing any tests to avoid making tests hang should a failure occur
    # on not all MPI ranks
    nptslocal = len(localpointcoords)
    nptsglobal = MPI.COMM_WORLD.allreduce(nptslocal, op=MPI.SUM)
    # Get parent PETSc cell indices on current MPI rank
    localparentcellindices = np.copy(swarm.getField("DMSwarm_cellid"))
    swarm.restoreField("DMSwarm_cellid")

    # Tests

    # get custom fields on swarm - will fail if didn't get created
    for name, size, dtype in fields:
        f = swarm.getField(name)
        assert len(f) == size*nptslocal
        assert f.dtype == dtype
        swarm.restoreField(name)
    # Check comm sizes match
    assert plex.comm.size == swarm.comm.size
    # Check coordinate list and parent cell indices match
    assert len(localpointcoords) == len(localparentcellindices)
    # check local points are found in list of input points
    for p in localpointcoords:
        assert np.any(np.isclose(p, inputpointcoords))
    # check local points are correct local points given mesh
    # partitioning (but don't require ordering to be maintained)
    assert np.allclose(np.sort(inputlocalpointcoords), np.sort(localpointcoords))
    # Check methods for checking number of points on current MPI rank
    assert len(localpointcoords) == swarm.getLocalSize()
    # Check there are as many local points as there are local cells
    # (excluding ghost cells in the halo)
    assert len(localpointcoords) == parentmesh.cell_set.size
    # Check total number of points on all MPI ranks is correct
    # (excluding ghost cells in the halo)
    assert nptsglobal == len(inputpointcoords)
    assert nptsglobal == swarm.getSize()
    # Check the parent cell indexes match those in the parent mesh
    cell_indexes = parentmesh.cell_closure[:, -1]
    for index in localparentcellindices:
        assert np.any(index == cell_indexes)


@pytest.mark.parallel
def test_pic_swarm_in_plex_parallel(parentmesh):
    test_pic_swarm_in_plex(parentmesh)


@pytest.mark.parallel(nprocs=2)  # nprocs == total number of mesh cells
def test_pic_swarm_in_plex_2d_2procs():
    test_pic_swarm_in_plex(UnitSquareMesh(1, 1))


@pytest.mark.parallel(nprocs=3)  # nprocs > total number of mesh cells
def test_pic_swarm_in_plex_2d_3procs():
    test_pic_swarm_in_plex(UnitSquareMesh(1, 1))
