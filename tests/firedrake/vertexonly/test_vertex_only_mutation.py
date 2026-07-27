from firedrake import *
from firedrake.vertexonly_mutation import VertexOnlyMeshMutator
from firedrake.utils import IntType
from pyop2.mpi import MPI
from pytest_mpi.parallel_assert import parallel_assert
import numpy as np
import pytest

@pytest.fixture
def parent_mesh():
    return UnitSquareMesh(5, 5, quadrilateral=False)

# Utility Functions

def cell_midpoints(mesh, with_halos=False):
    """Create deterministic physical point locations: one point at the midpoint of each mesh cell.
    Setting `with_halos=False` ensures the function returns only midpoints that are owned by each rank."""
    V = VectorFunctionSpace(mesh, "DG", 0)
    x = Function(V).interpolate(SpatialCoordinate(mesh))

    data = x.dat.data_ro_with_halos if with_halos else x.dat.data_ro
    return data.reshape((-1, mesh.geometric_dimension))

def locate_points(mesh, points):
    """Get local data (reference coordinates + parent cells) from physical point coordinates."""
    parent_cells, refcoords, _ = mesh.locate_cells_ref_coords_and_dists(points)
    return np.asarray(parent_cells, dtype=int), np.asarray(refcoords, dtype=float)

def vom_state_by_particle_id(vom):
    """Collect and return data on locally owned VOM points as a dictionary keyed by particle ID."""
    n_owned = vom.cell_set.size
    particle_ids = vom._particle_ids.dat.data_ro.copy()
    coordinates = vom.coordinates.dat.data_ro.copy()
    reference_coordinates = vom.reference_coordinates.dat.data_ro.copy()
    parent_cells = vom.topology.cell_parent_cell_list[:n_owned].copy()

    return {
        int(particle_id): (
            coordinates[i],
            reference_coordinates[i],
            int(parent_cells[i]),
        )
        for i, particle_id in enumerate(particle_ids)
    }

# Tests
@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize(
    "with_halos",
    [False, True],
    ids=["owned-only", "with-halos"],
)
def test_commit_reference_state_updates_swarm_and_vom_state(parent_mesh, with_halos):
    """Verify that the VOM's state gets mutated."""

    if with_halos and parent_mesh.comm.size == 1:
        pytest.skip("halo VOM points are only relevant when running in parallel")

    points = cell_midpoints(parent_mesh, with_halos=with_halos)
    vom = VertexOnlyMesh(parent_mesh, points, redundant=False)
    mutator = VertexOnlyMeshMutator(vom)

    n_owned = vom.cell_set.size
    vom_to_swarm = vom.cell_closure[:n_owned, -1]

    # Force the cached property to exist before the update. 
    # Calling `commit_reference_state` should update this property's data in place.
    _ = vom.topology.cell_parent_cell_list

    old_topology_version = vom.topology._topology_version
    old_num_cells = vom.num_cells()

    # Use current VOM coordinates in reverse order
    new_points = vom.coordinates.dat.data_ro[::-1].copy()
    new_parent_cells, new_refcoords = locate_points(parent_mesh, new_points)

    mutator.commit_reference_state(new_parent_cells, new_refcoords)

    assert np.array_equal(
        vom.topology.cell_parent_cell_list[:n_owned],
        new_parent_cells,
    )
    assert np.allclose(
        vom.reference_coordinates.dat.data_ro,
        new_refcoords,
    )

    swarm = vom.topology_dm

    parentcellnum = swarm.getField("parentcellnum").ravel().copy()
    swarm.restoreField("parentcellnum")
    assert np.array_equal(parentcellnum[vom_to_swarm], new_parent_cells)

    cell_id_name = swarm.getCellDMActive().getCellID()
    plex_parent_cells = swarm.getField(cell_id_name).ravel().copy()
    swarm.restoreField(cell_id_name)

    expected_plex_parent_cells = parent_mesh.topology.cell_closure[
        new_parent_cells, -1
    ]
    assert np.array_equal(
        plex_parent_cells[vom_to_swarm],
        expected_plex_parent_cells,
    )

    refcoord = swarm.getField("refcoord").copy()
    swarm.restoreField("refcoord")
    assert np.allclose(refcoord[vom_to_swarm, :], new_refcoords)

    # Committing reference state should not rebuild or resize the VOM
    assert vom.topology._topology_version == old_topology_version
    assert vom.num_cells() == old_num_cells
    assert vom.cell_set.size == n_owned


# NOTE: this test relies on the tracking particles through the PID field and therefore relies on the PID field migration working properly. 
# This is tested and verified separately.
@pytest.mark.parallel([1, 3])
def test_rebuild_vom_no_absorption_no_rank_transfer_preserves_state(parent_mesh):
    """Verify that rebuilding the VOM without absorption preserves particle state, that is
    we get the same set of particles and each particle ID remains associated 
    with the same physical coordinates, reference coordinates, and parent cell 
    regardless of any change in VOM ordering.
    """
    points = cell_midpoints(parent_mesh, with_halos=False)
    vom = VertexOnlyMesh(parent_mesh, points, redundant=False)
    mutator = VertexOnlyMeshMutator(vom)

    # Give each particle a changed but valid state. 
    # Reversing locally owned midpoints keeps the particles on cells owned by the same rank.
    new_coordinates = vom.coordinates.dat.data_ro[::-1].copy()
    new_parent_cells, new_reference_coordinates = locate_points(parent_mesh, new_coordinates)
    
    # NOTE: We write the updated state onto the VOM BEFORE rebuilding
    vom.coordinates.dat.data[:] = new_coordinates
    mutator.commit_reference_state(
        new_parent_cells,
        new_reference_coordinates,
    )

    expected_state = vom_state_by_particle_id(vom)
    old_local_count = vom.cell_set.size
    old_global_count = parent_mesh.comm.allreduce(old_local_count, op=MPI.SUM)
    old_version = vom.topology._topology_version

    mutator.rebuild_vom()

    actual_state = vom_state_by_particle_id(vom)
    new_global_count = parent_mesh.comm.allreduce(vom.cell_set.size, op=MPI.SUM)

    parallel_assert(
        expected_state.keys() == actual_state.keys(),
        "Particle IDs changed during a rebuild without absorption",
    )

    state_is_preserved = all(
        np.allclose(actual_state[pid][0], expected_state[pid][0])  # coords
        and np.allclose(actual_state[pid][1], expected_state[pid][1])  # ref coords
        and actual_state[pid][2] == expected_state[pid][2]  # parent cells
        for pid in expected_state
    )

    parallel_assert(
        state_is_preserved,
        "Particle state was not preserved during rebuilding",
    )

    parallel_assert(vom.cell_set.size == old_local_count)
    assert new_global_count == old_global_count
    assert vom.cell_set.size == vom.num_vertices()  # no halos in the new VOM

    new_version = vom.topology._topology_version
    assert new_version == old_version + 1
    assert new_version in vom.topology._topology_step_sfs


@pytest.mark.parallel(nprocs=3)
def test_rebuild_vom_no_absorption_migrates_particles(parent_mesh):
    owned_points = cell_midpoints(parent_mesh, with_halos=False)
    all_points = cell_midpoints(parent_mesh, with_halos=True)

    vom = VertexOnlyMesh(parent_mesh, owned_points, redundant=False)
    mutator = VertexOnlyMeshMutator(vom)

    n_parent_cells_owned = parent_mesh.cell_set.size
    ghost_cells = np.arange(n_parent_cells_owned, parent_mesh.num_vertices())

    parallel_assert(
        len(ghost_cells) > 0 and vom.cell_set.size > 0,
        "Every rank is expected to have a particle and a halo cell",
    )

    # Move the first particle on each rank into a cell owned by another rank
    new_coords = vom.coordinates.dat.data_ro.copy()
    new_coords[0] = all_points[ghost_cells[0]]

    new_parent_cells, new_refcoords = locate_points(parent_mesh, new_coords)

    # Determine the rank that owns the ghost cell
    _, sf_leaves, sf_remotes = parent_mesh.topology_dm.getPointSF().getGraph()
    leaf_owners = {
        int(leaf): int(remote[0])
        for leaf, remote in zip(sf_leaves, sf_remotes)
    }

    new_plex_parent_cell_id = parent_mesh.topology.cell_closure[new_parent_cells[0], -1]

    destination_rank = leaf_owners.get(int(new_plex_parent_cell_id))

    parallel_assert(destination_rank != parent_mesh.comm.rank)

    # Check where each particle goes: only particle 0 should move, the others should remain on their ranks
    particle_ids = vom._particle_ids.dat.data_ro.copy()

    expected_ranks = np.full(vom.cell_set.size, parent_mesh.comm.rank, dtype=int)
    expected_ranks[0] = destination_rank

    # lists particles originating on this rank before migration
    expected_state_pre_rebuild = [(int(pid), int(destination_rank), coord.copy())
                      for pid, destination_rank, coord in zip(particle_ids, expected_ranks, new_coords)]

    gathered_expected_state_pre_rebuild = parent_mesh.comm.allgather(expected_state_pre_rebuild)
    
    expected_state_post_rebuild = {
        pid: coord
        for rank_expected_state in gathered_expected_state_pre_rebuild
        for pid, destination_rank, coord in rank_expected_state
        if destination_rank == parent_mesh.comm.rank
    }

    # Commit state
    vom.coordinates.dat.data_wo[:] = new_coords
    mutator.commit_reference_state(new_parent_cells, new_refcoords)

    # Rebuild the VOM
    mutator.rebuild_vom()

    # Check state after rebuilding
    pids_post_rebuild = vom._particle_ids.dat.data_ro.copy()
    coords_post_rebuild = vom.coordinates.dat.data_ro.copy()

    state_post_rebuild = {
        int(pid): coord.copy()
        for pid, coord in zip (pids_post_rebuild, coords_post_rebuild)
    }

    parallel_assert(
        state_post_rebuild.keys() == expected_state_post_rebuild.keys(),
        "Particles did not migrate to the expected ranks"
    )

    parallel_assert(
        all(
            np.allclose(state_post_rebuild[pid], expected_state_post_rebuild[pid])
            for pid in expected_state_post_rebuild
        ),
        "Particle coordinates changed during migration"
    )

@pytest.mark.parallel([1, 3])
def test_rebuild_vom_with_absorption(parent_mesh):
    """Verify that absorbed VOM indices disappear and that the VOM is correctly resized."""
    points = cell_midpoints(parent_mesh, with_halos=False)
    vom = VertexOnlyMesh(parent_mesh, points, redundant=False)
    mutator = VertexOnlyMeshMutator(vom)

    state_pre_rebuild = vom_state_by_particle_id(vom)
    pids_pre_rebuild = vom._particle_ids.dat.data_ro.copy()

    # Counts
    local_count_pre_rebuild = vom.cell_set.size
    global_count_pre_rebuild = parent_mesh.comm.allreduce(local_count_pre_rebuild, op=MPI.SUM)
 
    # Remove every other particle on each rank (by construction, no cross-rank exchange occurs here)
    absorbed_local_indices = np.arange(0, vom.cell_set.size, 2, dtype=int)
    absorbed_pids = {int(pids_pre_rebuild[idx]) for idx in absorbed_local_indices}

    expected_state = {
        pid: state
        for pid, state in state_pre_rebuild.items()
        if pid not in absorbed_pids
    }

    global_absorbed_count = parent_mesh.comm.allreduce(len(absorbed_pids), op=MPI.SUM)

    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    # State + counts post rebuild
    state_post_rebuild = vom_state_by_particle_id(vom)
    local_count_post_rebuild = vom.cell_set.size
    global_count_post_rebuild = parent_mesh.comm.allreduce(local_count_post_rebuild, op=MPI.SUM)

    # Check particle identity
    parallel_assert(
        state_post_rebuild.keys() == expected_state.keys(),
        "VOM points post rebuild don't match expected survivor points"
    )

    parallel_assert(
        absorbed_pids.isdisjoint(state_post_rebuild.keys()),
        "Unexpected absorbed point found in VOM"
    )

    # Check particle state
    survivor_state_preserved = all(
        np.allclose(state_post_rebuild[pid][0], expected_state[pid][0])
        and np.allclose(state_post_rebuild[pid][1], expected_state[pid][1])
        and state_post_rebuild[pid][2] == expected_state[pid][2]
        for pid in expected_state
    )

    parallel_assert(survivor_state_preserved)

    # Check resizing
    parallel_assert(
        local_count_post_rebuild == local_count_pre_rebuild - len(absorbed_pids),
        "The local VOM size post rebuild does not match the expect local count"
    )

    assert(global_count_post_rebuild == global_count_pre_rebuild - global_absorbed_count)
    
@pytest.mark.parallel([1, 3])
def test_rebuild_vom_produces_empty_vom(parent_mesh):
    points = cell_midpoints(parent_mesh, with_halos=False)
    vom = VertexOnlyMesh(parent_mesh, points, redundant=False)
    mutator = VertexOnlyMeshMutator(vom)

    old_local_count = vom.cell_set.size
    old_topology_version = vom.topology._topology_version

    # Remove all points on all ranks
    absorbed_local_indices = np.arange(old_local_count, dtype=int)

    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)
    
    # Check that the rebuild completes
    parallel_assert(
        vom.cell_set.size == 0,
        "Unexpected particles found on some rank when VOM is supposed to be empty."
    )

    parallel_assert(
        vom.topology._topology_version == old_topology_version + 1,
        "The VOM topology version was not incremented"
    )

    # Check that the point SF of the mutated swarm is empty
    point_sf = vom.topology_dm.getPointSF()
    nroots, ilocal, iremote = point_sf.getGraph()
    
    nleaves = 0 if iremote is None else len(iremote)

    parallel_assert(
    vom.topology_dm.getLocalSize() == 0,
        "The swarm still contains particles",
    )

    parallel_assert(
        nroots == 0,
        "The empty swarm point SF has roots",
    )

    parallel_assert(
        nleaves == 0,
        "The empty swarm point SF has leaves",
    )


@pytest.mark.parallel([1, 3])
def test_rebuild_vom_invalid_absorbed_index_raises_value_error(parent_mesh):
    points = cell_midpoints(parent_mesh, with_halos=False)
    vom = VertexOnlyMesh(parent_mesh, points, redundant=False)
    mutator = VertexOnlyMeshMutator(vom)

    old_local_count = vom.cell_set.size
    old_topology_version = vom.topology._topology_version

    invalid_absorbed_local_idx = np.asarray([old_local_count + 1])

    with pytest.raises(ValueError):
        mutator.rebuild_vom(absorbed_vom_indices=invalid_absorbed_local_idx)

    # Verify that the rebuild doesn't complete
    parallel_assert(
        vom.cell_set.size == old_local_count,
        "The VOM changed despite a failed rebuild due to an invalid absorbed particle ID"
    )

    parallel_assert(
        vom.topology._topology_version == old_topology_version,
        "The VOM topology version changed despite a failed rebuild due to an invalid absorbed particle ID"
    )


@pytest.mark.parallel([1, 3])
def test_rebuild_vom_topology_step_sf_maps_old_to_new_ordering(parent_mesh):
    points = cell_midpoints(parent_mesh, with_halos=False)
    vom = VertexOnlyMesh(parent_mesh, points, redundant=False)
    mutator = VertexOnlyMeshMutator(vom)

    old_local_count = vom.cell_set.size
    offset = (parent_mesh.comm.scan(old_local_count, op=MPI.SUM) - old_local_count)

    root_values = np.arange(offset, offset + old_local_count, dtype=IntType)

    # Produce a rebuild that changes the VOM's ordering - e.g., reverse local coordinates
    new_coords = vom.coordinates.dat.data_ro[::-1].copy()
    new_parent_cells, new_refcoords = locate_points(parent_mesh, new_coords)

    # Absorb points: the step SF should have fewer leaves than roots
    absorbed_local_indices = np.arange(0, old_local_count, 3, dtype=int)

    survivors = np.ones(old_local_count, dtype=bool)
    survivors[absorbed_local_indices] = False

    # Build an expected VOM state using parent cells
    # Since the VOM points are cell centroids, parent cell IDs can be used to uniquely identify these VOM points
    expected_state_by_parent_cell = {
        int(new_parent_cells[i]): int(root_values[i])
        for i in np.where(survivors)[0]
    }

    # Commit the new state
    vom.coordinates.dat.data_wo[:] = new_coords
    mutator.commit_reference_state(new_parent_cells, new_refcoords)

    old_version = vom.topology._topology_version

    # Rebuild the VOM around the new state
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    new_version = vom.topology._topology_version
    parallel_assert(new_version == old_version + 1)
    parallel_assert(new_version in vom.topology._topology_step_sfs, "No step SF was stored for this rebuild")

    step_sf = vom.topology._topology_step_sfs[new_version]

    # Check properties of the step SF
    nroots, ilocal, iremote = step_sf.getGraph()
    nleaves = 0 if iremote is None else len(iremote)

    parallel_assert(nroots == old_local_count)
    parallel_assert(nleaves == vom.cell_set.size)

    # Broadcast through the step SF
    leaf_values = np.full(vom.cell_set.size, -1, dtype=IntType)

    mpi_type = MPI._typedict[root_values.dtype.char]
    root_values = np.ascontiguousarray(root_values)

    step_sf.bcastBegin(
        mpi_type,
        root_values,
        leaf_values,
        MPI.REPLACE
    )

    step_sf.bcastEnd(
        mpi_type,
        root_values,
        leaf_values,
        MPI.REPLACE
    )

    # Construct the expected leaf values array using the expected ordering
    parent_cells_post_rebuild = vom.topology.cell_parent_cell_list[:vom.cell_set.size]

    parallel_assert(
        all(
            int(cell) in expected_state_by_parent_cell
            for cell in parent_cells_post_rebuild
        ),
        "The rebuilt VOM contains an unexpected parent cell",
    )

    expected_leaf_values = np.asarray(
        [
            expected_state_by_parent_cell.get(int(cell))
            for cell in parent_cells_post_rebuild
        ],
        dtype=IntType
    )

    parallel_assert(np.array_equal(leaf_values, expected_leaf_values))


@pytest.mark.parallel([1, 3])
def test_successive_vom_rebuilds(parent_mesh):
    points = cell_midpoints(parent_mesh, with_halos=False)
    vom = VertexOnlyMesh(parent_mesh, points, redundant=False)
    mutator = VertexOnlyMeshMutator(vom)

    # Version 0
    v0 = vom.topology._topology_version
    local_count_v0 = vom.cell_set.size
    global_count_v0 = parent_mesh.comm.allreduce(local_count_v0, op=MPI.SUM)

    # We want to absorb 2 particles on each rank in two successive VOM rebuilds 
    # so we first ensure there are enough points on each rank
    parallel_assert(local_count_v0 >= 3)

    absorbed_indices_1 = np.asarray([0], dtype=int) # absorb first local particle
    pids_0 = vom._particle_ids.dat.data_ro.copy()
    absorbed_pid_1 = pids_0[absorbed_indices_1[0]] # get pid of absorbed particle

    # First rebuild
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_indices_1)

    v1 = vom.topology._topology_version
    local_count_v1 = vom.cell_set.size

    parallel_assert(v1 == v0 + 1)
    parallel_assert(local_count_v1 == local_count_v0 - 1)
    parallel_assert(v1 in vom.topology._topology_step_sfs)

    step_sf_1 = vom.topology._topology_step_sfs[v1]
    nroots_1, ilocal_1, iremote_1 = step_sf_1.getGraph()
    nleaves_1 = 0 if iremote_1 is None else len(iremote_1)

    parallel_assert(nroots_1 == local_count_v0)
    parallel_assert(nleaves_1 == local_count_v1)
    
    absorbed_indices_2 = np.asarray([0], dtype=int) # absorb first local particle
    pids_1 = vom._particle_ids.dat.data_ro.copy()
    absorbed_pid_2 = pids_1[absorbed_indices_2[0]] # get pid of absorbed particle

    # Second rebuild
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_indices_2)

    v2 = vom.topology._topology_version
    local_count_v2 = vom.cell_set.size

    parallel_assert(v2 == v1 + 1)
    parallel_assert(local_count_v2 == local_count_v1 - 1)
    parallel_assert(v2 in vom.topology._topology_step_sfs)

    step_sf_2 = vom.topology._topology_step_sfs[v2]
    nroots_2, ilocal_2, iremote_2 = step_sf_2.getGraph()
    nleaves_2 = 0 if iremote_2 is None else len(iremote_2)

    parallel_assert(nroots_2 == local_count_v1)
    parallel_assert(nleaves_2 == local_count_v2)

    parallel_assert(v1 in vom.topology._topology_step_sfs)
    parallel_assert(v2 in vom.topology._topology_step_sfs)

    pids_2 = vom._particle_ids.dat.data_ro.copy()

    parallel_assert(absorbed_pid_1 not in pids_2)
    parallel_assert(absorbed_pid_2 not in pids_2)

    # Verify that the global topology has changed
    global_count_v2 = parent_mesh.comm.allreduce(local_count_v2, op=MPI.SUM)

    assert(global_count_v2 == global_count_v0 - 2*parent_mesh.comm.size)










