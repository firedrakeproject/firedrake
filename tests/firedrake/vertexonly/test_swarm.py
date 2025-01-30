from firedrake import *
from firedrake.__future__ import *
from firedrake.utils import IntType, RealType
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
    local_midpoints = f.dat.data_ro.reshape(num_cells_local, m.geometric_dimension())
    local_midpoints_size = np.array(local_midpoints.size)
    local_midpoints_sizes = np.empty(MPI.COMM_WORLD.size, dtype=int)
    MPI.COMM_WORLD.Allgatherv(local_midpoints_size, local_midpoints_sizes)
    midpoints = np.empty((num_cells, m.geometric_dimension()), dtype=local_midpoints.dtype)
    MPI.COMM_WORLD.Allgatherv(local_midpoints, (midpoints, local_midpoints_sizes))
    assert len(np.unique(midpoints, axis=0)) == len(midpoints)
    return midpoints, local_midpoints


def cell_ownership(m):
    """Determine the MPI rank that the local partition thinks "owns" each cell
    in the mesh.

    :param m: The mesh to generate cell ownership for.

    :returns: A numpy array of MPI ranksl, indexed by cell number as given by
    m.locate_cell(point).

    """
    m.init()
    # Interpolating Constant(parent_mesh.comm.rank) into P0DG cleverly creates
    # a Function whose dat contains rank ownership information in an ordering
    # that is accessible using Firedrake's cell numbering. This is because, on
    # each rank, parent_mesh.comm.rank creates a Constant with the local rank
    # number, and halo exchange ensures that this information is visible, as
    # nessesary, to other processes.
    P0DG = FunctionSpace(m, "DG", 0)
    return assemble(interpolate(Constant(m.comm.rank), P0DG)).dat.data_ro_with_halos


def point_ownership(m, points, localpoints):
    """Determine the MPI rank that the local partition thinks "owns" the given
    points array.

    If the points are not in the local mesh partition or the halo, then the
    returned rank will be -1.

    :param m: The mesh to generate point ownership for.
    :param points: A numpy array of all points across all MPI ranks.
    :param localpoints: A numpy array of points that are known to be rank-local.

    :returns: A numpy array of MPI ranks, indexed by point number as given by
    m.locate_cell(point).

    """
    out_of_mesh_point = np.full((1, m.geometric_dimension()), np.inf)
    cell_numbers = np.empty(len(localpoints), dtype=int)
    i = 0
    for point in points:
        if any(np.array_equal(point, localpoint) for localpoint in localpoints):
            cell_numbers[i] = m.locate_cell(point)
            i += 1
        else:
            # need to still call locate_cell since it's collective
            m.locate_cell(out_of_mesh_point)
    # shouldn't find any Nones: all points should be in the local mesh partition
    assert all(cell_numbers != None)  # noqa: E711
    ownership = cell_ownership(m)
    return ownership[cell_numbers]


@pytest.fixture(params=["interval",
                        "square",
                        "squarequads",
                        "extruded",
                        pytest.param("extrudedvariablelayers", marks=pytest.mark.skip(reason="Extruded meshes with variable layers not supported and will hang when created in parallel")),
                        "cube",
                        "tetrahedron",
                        "immersedsphere",
                        "immersedsphereextruded",
                        "periodicrectangle",
                        "shiftedmesh"])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "squarequads":
        return UnitSquareMesh(2, 2, quadrilateral=True)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(2, 2), 3)
    elif request.param == "extrudedvariablelayers":
        return ExtrudedMesh(UnitIntervalMesh(3), np.array([[0, 3], [0, 3], [0, 2]]), np.array([3, 3, 2]))
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        m = UnitIcosahedralSphereMesh()
        m.init_cell_orientations(SpatialCoordinate(m))
        return m
    elif request.param == "immersedsphereextruded":
        m = UnitIcosahedralSphereMesh()
        m.init_cell_orientations(SpatialCoordinate(m))
        m = ExtrudedMesh(m, 3, extrusion_type="radial")
        return m
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


@pytest.fixture(params=["exclude_halos", "include_halos"])
def exclude_halos(request):
    if request.param == "exclude_halos":
        return True
    else:
        return False


# pic swarm tests

def test_pic_swarm_in_mesh(parentmesh, redundant, exclude_halos):
    """Generate points in cell midpoints of mesh `parentmesh` and check correct
    swarm is created in plex."""

    if not exclude_halos and parentmesh.comm.size == 1:
        pytest.skip("Testing halo behaviour in serial isn't worth the time")

    # Setup

    parentmesh.init()
    inputpointcoords, inputlocalpointcoords = cell_midpoints(parentmesh)
    inputcoordindices = np.arange(len(inputpointcoords))
    inputlocalpointcoordranks = point_ownership(parentmesh, inputpointcoords, inputlocalpointcoords)
    plex = parentmesh.topology.topology_dm
    from firedrake.petsc import PETSc
    other_fields = [("fieldA", 1, PETSc.IntType), ("fieldB", 2, PETSc.ScalarType)]

    if redundant:
        if MPI.COMM_WORLD.size == 1:
            pytest.skip("Testing redundant in serial isn't worth the time")
        # check redundant argument broadcasts from rank 0 by only supplying the
        # global cell midpoints only on rank 0. Note that this is the default
        # behaviour so it needn't be specified explicitly.
        if MPI.COMM_WORLD.rank == 0:
            swarm, original_swarm, n_missing_coords = mesh._pic_swarm_in_mesh(parentmesh, inputpointcoords, fields=other_fields, exclude_halos=exclude_halos)
        else:
            swarm, original_swarm, n_missing_coords = mesh._pic_swarm_in_mesh(parentmesh, np.empty(inputpointcoords.shape), fields=other_fields, exclude_halos=exclude_halos)
        input_rank = 0
        # inputcoordindices is the correct set of input indices for
        # redundant==True but I need to work out where they will be after
        # immersion in the parent mesh. I've done this by manually finding the
        # indices of inputpointcoords which are close to the
        # inputlocalpointcoords (this is nasty but I can't think of a better
        # way to do it!)
        indices_to_use = np.full(len(inputpointcoords), False)
        for lp in inputlocalpointcoords:
            for i, p in enumerate(inputpointcoords):
                if np.allclose(p, lp):
                    indices_to_use[i] = True
                    break
        input_local_coord_indices = inputcoordindices[indices_to_use]
    else:
        # When redundant == False we expect the same behaviour by only
        # supplying the local cell midpoints on each MPI ranks. Note that this
        # is not the default behaviour so it must be specified explicitly.
        swarm, original_swarm, n_missing_coords = mesh._pic_swarm_in_mesh(parentmesh, inputlocalpointcoords, fields=other_fields, redundant=redundant, exclude_halos=exclude_halos)
        input_rank = parentmesh.comm.rank
        input_local_coord_indices = np.arange(len(inputlocalpointcoords))

    have_halos = len(parentmesh.coordinates.dat.data_ro_with_halos) > len(parentmesh.coordinates.dat.data_ro)
    # collect from all ranks
    have_halos = MPI.COMM_WORLD.allreduce(have_halos, op=MPI.SUM)
    if not have_halos and not exclude_halos:
        # We can treat this as though we set exclude_halos=True. This should
        # happen in parallel with a mesh with 1 cell.
        exclude_halos = True

    # Get point coords on current MPI rank
    localpointcoords = np.copy(swarm.getField("DMSwarmPIC_coor").ravel())
    swarm.restoreField("DMSwarmPIC_coor")
    if len(inputpointcoords.shape) > 1:
        localpointcoords = np.reshape(localpointcoords, (-1, inputpointcoords.shape[1]))
    # Turn this into a number of points locally and MPI globally before
    # doing any tests to avoid making tests hang should a failure occur
    # on not all MPI ranks
    nptslocal = len(localpointcoords)
    nptsglobal = MPI.COMM_WORLD.allreduce(nptslocal, op=MPI.SUM)
    # Get parent PETSc cell indices on current MPI rank
    localparentcellindices = np.copy(swarm.getField("DMSwarm_cellid").ravel())
    swarm.restoreField("DMSwarm_cellid")

    # also get the global coordinate numbering
    globalindices = np.copy(swarm.getField("globalindex").ravel())
    swarm.restoreField("globalindex")

    # Tests

    # Since we have specified points at cell midpoints, we should have no
    # missing points
    assert n_missing_coords == 0

    # get custom fields on swarm - will fail if didn't get created
    for name, size, dtype in other_fields:
        f = swarm.getField(name).ravel()
        assert len(f) == size*nptslocal
        assert f.dtype == dtype
        swarm.restoreField(name)
    # Check comm sizes match
    assert plex.comm.size == swarm.comm.size
    # Check swarm fields are correct
    default_fields = [
        ("DMSwarmPIC_coor", parentmesh.geometric_dimension(), RealType),
        ("DMSwarm_cellid", 1, IntType),
        ("DMSwarm_rank", 1, IntType),
    ]
    default_extra_fields = [
        ("parentcellnum", 1, IntType),
        ("refcoord", parentmesh.topological_dimension(), RealType),
        ("globalindex", 1, IntType),
        ("inputrank", 1, IntType),
        ("inputindex", 1, IntType),
    ]
    if parentmesh.extruded:
        default_extra_fields.append(("parentcellbasenum", 1, IntType))
        default_extra_fields.append(("parentcellextrusionheight", 1, IntType))

    all_fields = default_fields + default_extra_fields + other_fields
    assert swarm.fields == all_fields
    assert swarm.default_fields == default_fields
    assert swarm.default_extra_fields == default_extra_fields
    assert swarm.other_fields == other_fields

    # Check coordinate list and parent cell indices match
    assert len(localpointcoords) == len(localparentcellindices)
    # check local points are found in list of input points
    for p in localpointcoords:
        assert np.any(np.isclose(p, inputpointcoords))
    if exclude_halos:
        # check local points are correct local points given mesh
        # partitioning (but don't require ordering to be maintained)
        assert np.allclose(np.sort(inputlocalpointcoords, axis=0),
                           np.sort(localpointcoords, axis=0))
    elif parentmesh.comm.size > 1:
        # If we have any points, there should be more than the input points
        if len(localpointcoords):
            assert len(localpointcoords) > len(inputlocalpointcoords)
        else:
            # otherwise there should be none
            assert len(localpointcoords) == len(inputlocalpointcoords)
    # Check methods for checking number of points on current MPI rank
    assert len(localpointcoords) == swarm.getLocalSize()
    if not parentmesh.extruded:
        if exclude_halos:
            # Check there are as many local points as there are local cells
            # (excluding ghost cells in the halo). This won't be true for extruded
            # meshes as the cell_set.size is the number of base mesh cells.
            assert len(localpointcoords) == parentmesh.cell_set.size
        elif parentmesh.comm.size > 1:
            # parentmesh.cell_set.total_size is the sum of owned and halo
            # points. We have a point in each cell, hence the below.
            assert len(localpointcoords) == parentmesh.cell_set.total_size
    else:
        if parentmesh.variable_layers:
            pytest.skip("Don't know how to calculate number of cells for variable layers")
        elif exclude_halos:
            ncells = parentmesh.cell_set.size * (parentmesh.layers - 1)
        else:
            ncells = parentmesh.cell_set.total_size * (parentmesh.layers - 1)
        assert len(localpointcoords) == ncells
    if exclude_halos:
        # Check total number of points on all MPI ranks is correct
        # (excluding ghost cells in the halo)
        assert nptsglobal == len(inputpointcoords)
    elif parentmesh.comm.size > 1:
        # If there are any points, there should be more than the input points
        if nptsglobal:
            assert nptsglobal > len(inputpointcoords)
        else:
            # otherwise there should be none
            assert nptsglobal == len(inputpointcoords)
    assert nptsglobal == swarm.getSize()

    # Check the parent cell indexes match those in the parent mesh
    cell_indexes = parentmesh.cell_closure[:, -1]
    for index in localparentcellindices:
        assert np.any(index == cell_indexes)

    # since we know all points are in the mesh, we can check that the global
    # indices are correct (i.e. they should be in rank order)
    allglobalindices = np.concatenate(parentmesh.comm.allgather(globalindices))
    if exclude_halos:
        assert np.array_equal(inputcoordindices, allglobalindices)
        _, idxs = np.unique(allglobalindices, return_index=True)
        assert len(idxs) == len(allglobalindices)
    else:
        assert np.array_equal(inputcoordindices, np.unique(allglobalindices))

    # Check that the rank numbering is correct. Since we know all points are at
    # the midpoints of cells, there should be no disagreement about cell
    # ownership and the voting algorithm should have no effect.
    owned_ranks = np.copy(swarm.getField("DMSwarm_rank").ravel())
    swarm.restoreField("DMSwarm_rank")
    if exclude_halos:
        assert np.array_equal(owned_ranks, inputlocalpointcoordranks)
    elif parentmesh.comm.size > 1:
        # The input ranks should be a subset of the owned ranks on the swarm
        assert np.all(np.isin(inputlocalpointcoordranks, owned_ranks))

    # check that the input rank is correct
    input_ranks = np.copy(swarm.getField("inputrank").ravel())
    swarm.restoreField("inputrank")
    if exclude_halos:
        assert np.all(input_ranks == input_rank)
    elif parentmesh.comm.size > 1:
        # The input rank should be a within the input ranks array on the swarm
        # and we shouldn't have ranks which are greater than the comm size
        if len(input_ranks):
            assert np.isin(input_rank, input_ranks)
        assert np.all(input_ranks < parentmesh.comm.size)

    # check that the input index is correct
    input_indices = np.copy(swarm.getField("inputindex").ravel())
    swarm.restoreField("inputindex")
    if exclude_halos:
        assert np.array_equal(input_indices, input_local_coord_indices)
        if redundant:
            assert np.array_equal(input_indices, globalindices)
    elif parentmesh.comm.size > 1:
        # The input indices should be a subset of the indices on the swarm
        assert np.all(np.isin(input_local_coord_indices, input_indices))
        if redundant:
            assert np.array_equal(np.unique(input_indices), np.sort(globalindices))

    # check we have unique parent cell numbers, which we should since we have
    # points at cell midpoints
    parentcellnums = np.copy(swarm.getField("parentcellnum").ravel())
    swarm.restoreField("parentcellnum")
    assert len(np.unique(parentcellnums)) == len(parentcellnums)

    # Now have DMPLex compute the cell IDs in cases where it can:
    if (
        parentmesh.coordinates.ufl_element().family() != "Discontinuous Lagrange"
        and parentmesh.geometric_dimension() == parentmesh.topological_dimension()
        and not parentmesh.extruded
        and not parentmesh.coordinates.dat.dat_version > 0  # shifted mesh
    ):
        swarm.setPointCoordinates(localpointcoords, redundant=False,
                                  mode=PETSc.InsertMode.INSERT_VALUES)
        petsclocalparentcellindices = np.copy(swarm.getField("DMSwarm_cellid").ravel())
        swarm.restoreField("DMSwarm_cellid")
        if exclude_halos:
            assert np.all(petsclocalparentcellindices == localparentcellindices)
        elif parentmesh.comm.size > 1:
            # setPointCoordinates doesn't let us have points in halos so we
            # have to check for a subset
            assert np.all(np.isin(petsclocalparentcellindices, localparentcellindices))

    # check original swarm has correct properties
    assert original_swarm.fields != swarm.fields  # We don't currently rearrange custom fields
    assert original_swarm.default_fields == swarm.default_fields
    assert original_swarm.default_extra_fields == swarm.default_extra_fields
    assert original_swarm.other_fields != swarm.other_fields
    assert isinstance(original_swarm.getCellDM(), PETSc.DMSwarm)

    # out_of_mesh_point = np.full((2, parentmesh.geometric_dimension()), np.inf)
    # swarm, n_missing_coords = mesh._pic_swarm_in_mesh(parentmesh, out_of_mesh_point, fields=fields)
    # assert n_missing_coords == 2


@pytest.mark.parallel
def test_pic_swarm_in_mesh_parallel(parentmesh, redundant, exclude_halos):
    test_pic_swarm_in_mesh(parentmesh, redundant, exclude_halos)


@pytest.mark.parallel(nprocs=2)  # nprocs == total number of mesh cells
def test_pic_swarm_in_mesh_2d_2procs():
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=False, exclude_halos=True)
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=True, exclude_halos=True)
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=False, exclude_halos=True)
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=True, exclude_halos=True)


@pytest.mark.parallel(nprocs=3)  # nprocs > total number of mesh cells
def test_pic_swarm_in_mesh_2d_3procs():
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=False, exclude_halos=True)
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=True, exclude_halos=True)
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=False, exclude_halos=False)
    test_pic_swarm_in_mesh(UnitSquareMesh(1, 1), redundant=True, exclude_halos=False)
