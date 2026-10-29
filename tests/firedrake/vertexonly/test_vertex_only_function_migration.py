from firedrake import *
from firedrake.utils import IntType
from firedrake.vertexonly_mutation import VertexOnlyMeshMutator
from pytest_mpi.parallel_assert import parallel_assert
import numpy as np
import pytest


@pytest.fixture
def parent_mesh():
    return UnitSquareMesh(5, 5, quadrilateral=False)


@pytest.fixture
def vom(request, parent_mesh):
    with_halos = getattr(request, "param", False)

    if with_halos and parent_mesh.comm.size == 1:
        pytest.skip("Halo inputs are only meaningful in parallel")

    points = cell_midpoints(parent_mesh, with_halos=with_halos)

    return VertexOnlyMesh(parent_mesh, points, redundant=False)


@pytest.fixture(params=["scalar-DG0", "vector-DG0", "tensor-DG0"])
def vom_fs(request, vom):
    if request.param == "scalar-DG0":
        return FunctionSpace(vom, "DG", 0)

    if request.param == "vector-DG0":
        return VectorFunctionSpace(vom, "DG", 0, dim=vom.geometric_dimension)

    if request.param == "tensor-DG0":
        return TensorFunctionSpace(vom, "DG", 0, shape=(vom.geometric_dimension, vom.geometric_dimension))

# Utility Functions

def cell_midpoints(mesh, with_halos=False):
    """Return the midpoint of each cell of the mesh, in the cell order of the mesh."""
    V = VectorFunctionSpace(mesh, "DG", 0)
    x = Function(V).interpolate(SpatialCoordinate(mesh))
    return (x.dat.data_ro_with_halos if with_halos else x.dat.data_ro).copy()


def locate_points(mesh, points):
    """Get local data (reference coordinates + parent cells) from physical point coordinates."""
    parent_cells, refcoords, _ = mesh.locate_cells_ref_coords_and_dists(points)
    return np.asarray(parent_cells, dtype=IntType), np.asarray(refcoords, dtype=float)


def assign_function_values_using_pids(pids, value_size, dtype):
    pids = np.asarray(pids)[:, None]
    components = 100 * np.arange(value_size)[None, :]
    return (pids + components + 0.25).astype(dtype)


@pytest.mark.parallel([1, 3])
def test_vom_functions_migrate_lazily(vom, vom_fs):
    f = Function(vom_fs)
    g = Function(vom_fs)

    old_f_topo_version = f._mesh_topology_version
    old_g_topo_version = g._mesh_topology_version

    old_vom_topo_version = vom.topology._topology_version
    parallel_assert(old_f_topo_version == old_vom_topo_version and old_g_topo_version == old_vom_topo_version,
                    "Functions did not start on the current VOM topology")

    mutator = VertexOnlyMeshMutator(vom)

    absorbed_local_indices = np.arange(0, vom.topology.cells.owned.local_size, 2, dtype=IntType)
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    new_vom_topo_version = vom.topology._topology_version

    # Verify that rebuilding the VOM does not eagerly migrate the Functions
    parallel_assert(
        f._mesh_topology_version == old_f_topo_version
        and g._mesh_topology_version == old_g_topo_version,
        "A Function migrated before its data was accessed",
    )

    # Only access one of the two Functins first
    _ = f.dat

    parallel_assert(f._mesh_topology_version == new_vom_topo_version, "Accessing the Function's data did not migrate it to the current VOM topology")
    parallel_assert(g._mesh_topology_version == old_g_topo_version, "A Function who's data wasn't accessed after the VOM's rebuild was unexpectedly migrated")

    # Access the other Function now and ensure it has migrated
    _ = g.dat
    parallel_assert(g._mesh_topology_version == new_vom_topo_version, "Accessing the Function's data did not migrate it to the current VOM topology")


# FIXME
@pytest.mark.parallel([1, 3])
def test_vom_functions_migrate_eagerly(vom, vom_fs):
    f = Function(vom_fs)
    g = Function(vom_fs)

    # Define a Function on a distinct FunctionSpace
    W = VectorFunctionSpace(vom, "DG", 0, dim=3)
    h = Function(W)

    old_f_topo_version = f._mesh_topology_version
    old_g_topo_version = g._mesh_topology_version
    old_h_topo_version = h._mesh_topology_version

    old_vom_topo_version = vom.topology._topology_version

    parallel_assert(old_f_topo_version == old_vom_topo_version
                    and old_g_topo_version == old_vom_topo_version
                    and h._mesh_topology_version == old_vom_topo_version,
                    "Functions did not start on the current VOM topology")

    mutator = VertexOnlyMeshMutator(vom)

    absorbed_local_indices = np.arange(0, vom.topology.cells.owned.local_size, 2, dtype=IntType)
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    new_vom_topo_version = vom.topology._topology_version

    # Neither Function should have migrated yet
    parallel_assert(
        f._mesh_topology_version == old_f_topo_version
        and g._mesh_topology_version == old_g_topo_version
        and h._mesh_topology_version == old_h_topo_version,
        "A Function migrated before eager migration was executed",
    )

    # Migrate all live Functions registered on this VOM
    vom.topology._migrate_all_functions()

    parallel_assert(
        f._mesh_topology_version == new_vom_topo_version
        and g._mesh_topology_version == new_vom_topo_version
        and h._mesh_topology_version == new_vom_topo_version,
        "Not all live Functions were migrated eagerly",
    )

    # The Functions retain their original FunctionSpace objects
    parallel_assert(
        f.function_space() is vom_fs
        and g.function_space() is vom_fs
        and h.function_space() is W,
        "Migrating a Function changed its underlying FunctionSpace object",
    )


@pytest.mark.parallel([1, 3])
def test_vom_function_migrates_once_per_vom_rebuild(vom, vom_fs):
    f = Function(vom_fs)

    old_f_topo_version = f._mesh_topology_version
    old_vom_topo_version = vom.topology._topology_version

    parallel_assert(old_f_topo_version == old_vom_topo_version,
                    "The Function did not start on the current VOM topology")

    mutator = VertexOnlyMeshMutator(vom)

    # Do one VOM rebuild: absorb one particle on each rank
    absorbed_local_indices = np.asarray([0])
    mutator.rebuild_vom(absorbed_local_indices)

    new_vom_topo_version = vom.topology._topology_version

    # Access the Function's data
    _ = f.dat

    parallel_assert(f._mesh_topology_version == new_vom_topo_version, "Accessing the Function's data did not migrate it to the current VOM topology")

    # Access the Function's data again
    _ = f.dat

    parallel_assert(f._mesh_topology_version == new_vom_topo_version, "Accessing the Function's data another time has changed its mesh topology version attribute")


@pytest.mark.parallel([1, 3])
def test_vom_function_correctly_migrates_under_absorbed_points(vom, vom_fs):
    f = Function(vom_fs)
    value_size = vom_fs.value_size

    pids_before_migration = vom._particle_ids.dat.data_ro.copy()
    old_dat = f.dat

    # Use the particle ID field as Function values
    f_vals_before_migration = assign_function_values_using_pids(pids_before_migration, value_size, old_dat.dtype)
    old_dat.data_wo.reshape(-1, value_size)[:] = f_vals_before_migration

    old_function_topo_version = f._mesh_topology_version
    old_local_count = vom.topology.cells.owned.local_size

    # Remove every other particle on each rank (by construction, no cross-rank exchange occurs here)
    mutator = VertexOnlyMeshMutator(vom)

    absorbed_local_indices = np.arange(0, vom.topology.cells.owned.local_size, 2, dtype=IntType)
    absorbed_pids = {int(pids_before_migration[i]) for i in absorbed_local_indices}

    expected_surviving_pids = {int(pid) for pid in pids_before_migration if pid not in absorbed_pids}
    expected_local_count = old_local_count - len(absorbed_local_indices)

    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    # Trigger Function migration through dat access
    new_dat = f.dat
    f_vals_after_migration = new_dat.data_ro.reshape(-1, value_size).copy()
    pids_after_migration = vom._particle_ids.dat.data_ro.copy()
    expected_f_vals = assign_function_values_using_pids(pids_after_migration, value_size, new_dat.dtype)

    actual_surviving_pids = {int(pid) for pid in pids_after_migration}

    function_migration_is_correct = (
        actual_surviving_pids == expected_surviving_pids
        and absorbed_pids.isdisjoint(actual_surviving_pids)
        and f_vals_after_migration.shape == (expected_local_count, value_size)
        and np.allclose(f_vals_after_migration, expected_f_vals)
        # -- check storage was correctly replaced (uses current FS layout)
        and new_dat is not old_dat
        and new_dat.axes is vom_fs.axes
        and new_dat.data_ro.shape[0] == vom.topology.cells.owned.local_size  # owned nodes match VOM
        and new_dat.data_ro.shape[1:] == vom_fs.shape
        and f.function_space() is vom_fs
        and f._mesh_topology_version == vom.topology._topology_version
        and f._mesh_topology_version == old_function_topo_version + 1
    )

    parallel_assert(
        function_migration_is_correct,
        "Function values were not migrated correctly after removing some points from the VOM",
    )


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize(
    "vom",
    [False, True],
    indirect=True,
    ids=["owned-points", "owned-and-halo-points"],
)
def test_vom_function_migrates_to_empty_vom_layout(vom, vom_fs):
    f = Function(vom_fs)

    # Remove every point on every rank
    mutator = VertexOnlyMeshMutator(vom)
    absorbed_local_indices = np.arange(0, vom.topology.cells.owned.local_size, dtype=IntType)
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    # Trigger Function migration through dat access
    new_dat = f.dat
    with new_dat.vec_ro as vec:
        n_local, n_global = vec.getLocalSize(), vec.getSize()

    parallel_assert(f._mesh_topology_version == vom.topology._topology_version, "the Function has not been migrated to the current topology")
    parallel_assert(new_dat.data_ro.shape == (0, *vom_fs.shape), "the owned data is not empty")
    parallel_assert(new_dat.data_ro_with_halos.shape == (0, *vom_fs.shape), "the halo data is not empty")
    parallel_assert(n_local == 0, "this rank still owns DoFs")
    parallel_assert(n_global == 0, "some rank still owns DoFs")


@pytest.mark.parallel(nprocs=3)
def test_vom_function_migrates_under_point_redistribution(vom, vom_fs):
    parent_mesh = vom._parent_mesh
    comm = parent_mesh.comm
    f = Function(vom_fs)
    mutator = VertexOnlyMeshMutator(vom)

    # Give every point a value that is computed from its particle ID
    pids_before = vom.topology._particle_ids.dat.data_ro.copy()
    n_owned_before = vom.topology.cells.owned.local_size
    values = assign_function_values_using_pids(pids_before, vom_fs.value_size, f.dat.dtype)
    f.dat.data_wo[:] = values.reshape((-1, *vom_fs.shape))

    # Move the first point on rank 0 to the midpoint of its first ghost cell
    # Onwed cells come first and the ghost cells follow them
    midpoints = cell_midpoints(parent_mesh, with_halos=True)
    new_coords = vom.coordinates.dat.data_ro.copy()
    destination_rank = None
    moved_pid = None

    if comm.rank == 0:
        target_cell = parent_mesh.cells.owned.local_size
        new_coords[0] = midpoints[target_cell]
        moved_pid = pids_before[0]

        _, sf_leaves, sf_remotes = parent_mesh.topology_dm.getPointSF().getGraph()
        target_plex_cell_id = parent_mesh.topology._new_to_old_cell_numbering[target_cell] # plex ID of the target ghost cell
        (destination_rank, ) = sf_remotes[sf_leaves == target_plex_cell_id, 0]
        destination_rank = int(destination_rank)

    # Tell all other ranks which particle moved and where
    destination_rank = comm.bcast(destination_rank, root=0)
    moved_pid = comm.bcast(moved_pid, root=0)
    parallel_assert(destination_rank != 0, "Rank 0 did not select a cell owned by another rank")

    # Commit new state + rebuild
    new_parent_cells, new_refcoords = locate_points(parent_mesh, new_coords)
    vom.coordinates.dat.data_wo[:] = new_coords
    mutator.commit_reference_state(new_parent_cells, new_refcoords)
    mutator.rebuild_vom()

    new_dat = f.dat  # trigger migration
    pids_after = vom._particle_ids.dat.data_ro.copy()
    expected_values = assign_function_values_using_pids(pids_after, vom_fs.value_size, new_dat.dtype)

    n_expected = n_owned_before - (comm.rank == 0) + (comm.rank == destination_rank)
    parallel_assert(vom.topology.cells.owned.local_size == n_expected, "the number of points on this rank is wrong")
    parallel_assert(new_dat.data_ro.shape == (n_expected, *vom_fs.shape), "the Dat does not match the new points")
    parallel_assert(
        np.allclose(new_dat.data_ro.reshape((n_expected, -1)), expected_values),
        "the Function values did not follow their points",
    )
    parallel_assert(
        (moved_pid in pids_after) == (comm.rank == destination_rank),
        "the moved point is not on the destination rank",
    )
    parallel_assert(f._mesh_topology_version == vom.topology._topology_version, "the Function is not on the current version")


@pytest.mark.parallel([1, 3])
def test_vom_function_migrates_under_successive_vom_rebuilds(vom, vom_fs):
    f = Function(vom_fs)
    value_size = vom_fs.value_size
    mutator = VertexOnlyMeshMutator(vom)
    v0 = vom.topology._topology_version
    n_owned_v0 = vom.topology.cells.owned.local_size
    f_version_v0 = f._mesh_topology_version
    f_dat_0 = f.dat

    parallel_assert(f_version_v0 == v0,
                    "The Function did not start on the current VOM topology")

    # We do successive VOM rebuilds, removing one particle from each rank at each rebuild
    # Hence we need at least 3 particles on each rank
    parallel_assert(n_owned_v0 >= 3, "The test requires at least three particles per rank")

    # Assign PID-dependent values to the Function so we're able to trace its values through the rebuilds
    pids_v0 = vom._particle_ids.dat.data_ro.copy()
    values_v0 = assign_function_values_using_pids(pids_v0, value_size, f.dat.dtype)
    f.dat.data_wo[:] = values_v0.reshape((-1, *vom_fs.shape))

    # First VOM rebuild: absorb the first particle on each rank
    absorbed_indices_v1 = np.asarray([0], dtype=IntType)
    absorbed_pids_v1 = pids_v0[0]
    mutator.rebuild_vom(absorbed_indices_v1)

    v1 = vom.topology._topology_version
    pids_v1 = vom._particle_ids.dat.data_ro.copy()

    parallel_assert(
        v1 == v0 + 1
        and f._mesh_topology_version == f_version_v0,
        "The Function migrated between the two VOM rebuilds",
    )

    # Define a new function
    g = Function(vom_fs)
    parallel_assert(g._mesh_topology_version == v1)

    # Second VOM rebuild: absorb the last particle on each rank
    absorbed_indices_v2 = np.asarray([vom.topology.cells.owned.local_size - 1], dtype=IntType)

    # NOTE: particle ids migrate eagerly during the rebuild hence the last integer in the PID array
    # is the ID of the last particle in the new numbering.
    absorbed_pids_v2 = pids_v1[-1]

    mutator.rebuild_vom(absorbed_indices_v2)

    v2 = vom.topology._topology_version
    pids_v2 = vom._particle_ids.dat.data_ro.copy()

    # Subtracting two from the original local count is correct here since all particles stay in their original ranks
    # across the two rebuilds.
    expected_local_count = n_owned_v0 - 2
    expected_surviving_pids = {pid for pid in pids_v0} - {absorbed_pids_v1, absorbed_pids_v2}

    # Since the Function hasn't been accessed yet it should still be two versions behind
    parallel_assert(v2 == v0 + 2 and f._mesh_topology_version == f_version_v0)

    # Access the Function now: must migrate through the composition of the two step SFs v2 -> v1 and v1 -> v0.
    new_dat = f.dat

    expected_values_v2 = assign_function_values_using_pids(pids_v2, value_size, new_dat.dtype)

    actual_surviving_pids = {pid for pid in pids_v2}

    migration_is_correct = (
        actual_surviving_pids == expected_surviving_pids
        and vom.topology.cells.owned.local_size == expected_local_count
        and new_dat.data_ro.shape == (expected_local_count, *vom_fs.shape)
        and np.allclose(new_dat.data_ro.reshape((-1, value_size)), expected_values_v2)
        and new_dat is not f_dat_0
        and f._mesh_topology_version == v2
    )

    parallel_assert(migration_is_correct, "The Function did not properly migrate to the latest VOM topology version")

    # Since Function g hasn't been accessed it should still display the VOM topology version it was originally created on
    parallel_assert(g._mesh_topology_version == v1)
