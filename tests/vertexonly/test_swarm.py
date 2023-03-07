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
    m.init()
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(SpatialCoordinate(m))
    # since mesh may be distributed, the number of cells on the MPI rank
    # may not be the same on all ranks (note we exclude ghost cells
    # hence using num_cells_local = m.cell_set.size). Below local means
    # MPI rank local.
    num_cells_local = len(f.dat.data_ro)
    num_cells = MPI.COMM_WORLD.allreduce(num_cells_local, op=MPI.SUM)
    # reshape is for 1D case where f.dat.data_ro has shape (num_cells_local,)
    local_midpoints = f.dat.data_ro.reshape(num_cells_local, m.ufl_cell().geometric_dimension())
    local_midpoints_size = np.array(local_midpoints.size)
    local_midpoints_sizes = np.empty(MPI.COMM_WORLD.size, dtype=int)
    MPI.COMM_WORLD.Allgatherv(local_midpoints_size, local_midpoints_sizes)
    midpoints = np.empty((num_cells, m.ufl_cell().geometric_dimension()), dtype=local_midpoints.dtype)
    MPI.COMM_WORLD.Allgatherv(local_midpoints, (midpoints, local_midpoints_sizes))
    assert len(np.unique(midpoints, axis=0)) == len(midpoints)
    return midpoints, local_midpoints


@pytest.fixture(params=["interval",
                        "square",
                        "extruded",
                        pytest.param("extrudedvariablelayers", marks=pytest.mark.skip(reason="Extruded meshes with variable layers not supported and will hang when created in parallel")),
                        "cube",
                        "tetrahedron",
                        pytest.param("immersedsphere", marks=pytest.mark.skip(reason="immersed parent meshes not supported and will segfault PETSc when creating the DMSwarm")),
                        "periodicrectangle",
                        "shiftedmesh"])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(2, 2), 3)
    elif request.param == "extrudedvariablelayers":
        return ExtrudedMesh(UnitIntervalMesh(3), np.array([[0, 3], [0, 3], [0, 2]]), np.array([3, 3, 2]))
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        return UnitIcosahedralSphereMesh()
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)
    elif request.param == "shiftedmesh":
        m = UnitSquareMesh(10, 10)
        m.coordinates.dat.data[:] -= 0.5
        return m


@pytest.fixture(params=["redundant", "nonredundant"])
def redundant(request):
    if request.param == "redundant":
        return True
    else:
        return False


# pic swarm tests

def test_pic_swarm_in_mesh(parentmesh, redundant):
    """Generate points in cell midpoints of mesh `parentmesh` and check correct
    swarm is created in plex."""

    # Setup
    parentmesh.init()
    # The coords dat version is > 0 for a shifted mesh. We need to save the it
    # here because a bug somewhere in the kernel generation of
    # MeshGeometry.locate_cell_and_reference_coordinate changes its value.
    # Accessing the coordinates ought to be a read only operation but, for some
    # reason, it increments the dat version.
    coords_dat_version = parentmesh.coordinates.dat.dat_version
    inputpointcoords, inputlocalpointcoords = cell_midpoints(parentmesh)
    plex = parentmesh.topology.topology_dm
    from firedrake.petsc import PETSc
    fields = [("fieldA", 1, PETSc.IntType), ("fieldB", 2, PETSc.ScalarType)]

    if redundant:
        # check redundant argument broadcasts from rank 0 by only supplying the
        # global cell midpoints only on rank 0. Note that this is the default
        # behaviour so it needn't be specified explicitly.
        if MPI.COMM_WORLD.rank == 0:
            swarm = mesh._pic_swarm_in_mesh(parentmesh, inputpointcoords, fields=fields)
        else:
            swarm = mesh._pic_swarm_in_mesh(parentmesh, np.empty(inputpointcoords.shape), fields=fields)
    else:
        # When redundant == False we expect the same behaviour by only
        # supplying the local cell midpoints on each MPI ranks. Note that this
        # is not the default behaviour so it must be specified explicitly.
        swarm = mesh._pic_swarm_in_mesh(parentmesh, inputlocalpointcoords, fields=fields, redundant=redundant)

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
    assert np.allclose(np.sort(inputlocalpointcoords, axis=0),
                       np.sort(localpointcoords, axis=0))
    # Check methods for checking number of points on current MPI rank
    assert len(localpointcoords) == swarm.getLocalSize()
    if not parentmesh.extruded:
        # Check there are as many local points as there are local cells
        # (excluding ghost cells in the halo). This won't be true for extruded
        # meshes as the cell_set.size is the number of base mesh cells.
        assert len(localpointcoords) == parentmesh.cell_set.size
    else:
        if parentmesh.variable_layers:
            ncells = sum(height - 1 for _, height in parentmesh.layers)
        else:
            ncells = parentmesh.cell_set.size * (parentmesh.layers - 1)
        assert len(localpointcoords) == ncells
    # Check total number of points on all MPI ranks is correct
    # (excluding ghost cells in the halo)
    assert nptsglobal == len(inputpointcoords)
    assert nptsglobal == swarm.getSize()
    # Check the parent cell indexes match those in the parent mesh unless
    # parent mesh is shifted, in which case they should all be -1
    cell_indexes = parentmesh.cell_closure[:, -1]
    for index in localparentcellindices:
        if coords_dat_version > 0:
            assert index == -1
        else:
            assert np.any(index == cell_indexes)

    # Now have DMPLex compute the cell IDs in cases where it can:
    if (
        parentmesh.coordinates.ufl_element().family() != "Discontinuous Lagrange"
        and not parentmesh.extruded
        and not coords_dat_version > 0
    ):
        swarm.setPointCoordinates(localpointcoords, redundant=False,
                                  mode=PETSc.InsertMode.INSERT_VALUES)
        petsclocalparentcellindices = np.copy(swarm.getField("DMSwarm_cellid"))
        swarm.restoreField("DMSwarm_cellid")
        assert np.all(petsclocalparentcellindices == localparentcellindices)


@pytest.mark.parallel
def test_pic_swarm_in_mesh_parallel(parentmesh, redundant):
    test_pic_swarm_in_mesh(parentmesh, redundant)


@pytest.mark.parallel(nprocs=2)  # nprocs == total number of mesh cells
def test_pic_swarm_in_mesh_2d_2procs():
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=False)
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=True)


@pytest.mark.parallel(nprocs=3)  # nprocs > total number of mesh cells
def test_pic_swarm_in_mesh_2d_3procs():
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=False)
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=False)
