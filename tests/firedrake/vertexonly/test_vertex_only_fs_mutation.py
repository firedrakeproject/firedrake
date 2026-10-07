from firedrake import *
from firedrake.vertexonly_mutation import VertexOnlyMeshMutator
import pytest
from pytest_mpi.parallel_assert import parallel_assert
import numpy as np
from mpi4py import MPI

@pytest.fixture
def parent_mesh():
    return UnitSquareMesh(5, 5, quadrilateral=False)


@pytest.fixture
def vom(parent_mesh):
    points = cell_midpoints(parent_mesh, with_halos=False)
    return VertexOnlyMesh(parent_mesh, points, redundant=False)


@pytest.fixture(params=["scalar", "vector", "tensor"])
def vom_fs(request, vom):
    if request.param == "scalar":
        return FunctionSpace(vom, "DG", 0)

    if request.param == "vector":
        return VectorFunctionSpace(vom, "DG", 0, dim=vom.geometric_dimension)

    if request.param == "tensor":
        return TensorFunctionSpace(vom, "DG", 0, shape=(2, 2))


def cell_midpoints(mesh, with_halos=False):
    """
    Create deterministic physical point locations-one point at the midpoint of each mesh cell-returned in the cell order.
    Setting `with_halos=False` ensures we only get midpoints that are owned by each rank."""
    V = VectorFunctionSpace(mesh, "DG", 0)
    x = Function(V).interpolate(SpatialCoordinate(mesh))

    data = x.dat.data_ro_with_halos if with_halos else x.dat.data_ro
    cell_nodes = V.cell_node_list[:, 0]

    if not with_halos:
        cell_nodes = cell_nodes[:mesh.cell_set.size]

    return data[cell_nodes].copy()


def locate_points(mesh, points):
    """Get local data (reference coordinates + parent cells) from physical point coordinates."""
    parent_cells, refcoords, _ = mesh.locate_cells_ref_coords_and_dists(points)
    return np.asarray(parent_cells, dtype=int), np.asarray(refcoords, dtype=float)


@pytest.mark.parallel([1, 3])
def test_vom_fs_data_refreshes_after_vom_mutation(vom, vom_fs):
    """Check that FunctionSpace caches get invalidated and recomputed after a VOM topology change."""
    mutator = VertexOnlyMeshMutator(vom)

    T = vom_fs.topological

    # Read topology-dependent properties from the caches
    old_shared_data = T._shared_data
    old_dof_dset = T.dof_dset
    old_global_numbering = T.global_numbering
    old_dm = T.dm
    old_ises = T._ises
    old_cell_node_list = T.cell_node_list
    old_cell_node_map = T.cell_node_map()

    # Mutate the VOM topologically by removing every other point on each rank
    absorbed_local_indices = np.arange(0, vom.cell_set.size, 2, dtype=int)
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    new_shared_data = T._shared_data
    new_dof_dset = T.dof_dset
    new_global_numbering = T.global_numbering
    new_dm = T.dm
    new_ises = T._ises
    new_cell_node_list = T.cell_node_list
    new_cell_node_map = T.cell_node_map()

    parallel_assert(new_shared_data is not old_shared_data)
    parallel_assert(new_dof_dset is not old_dof_dset)
    parallel_assert(new_global_numbering is not old_global_numbering)
    parallel_assert(new_dm is not old_dm)
    parallel_assert(new_ises is not old_ises)
    parallel_assert(new_cell_node_list is not old_cell_node_list)
    parallel_assert(new_cell_node_map is not old_cell_node_map)


@pytest.mark.parallel([1, 3])
def test_vom_fs_data_resizes_to_match_vom_topology(vom, vom_fs):
    T = vom_fs.topological
    mutator = VertexOnlyMeshMutator(vom)

    n_old_local_nodes = T.node_set.size  # 1 node per VOM point
    n_old_global_dofs = T.dim()  # total number of scalar DoFs across all ranks

    # Mutate the VOM topologically by removing every other point on each rank
    absorbed_local_indices = np.arange(0, vom.cell_set.size, 2, dtype=int)
    n_local_absorbed = len(absorbed_local_indices)
    n_global_absorbed = vom.comm.allreduce(n_local_absorbed, op=MPI.SUM)

    # Expected size post rebuild
    expected_n_local_nodes = n_old_local_nodes - n_local_absorbed
    expected_n_global_dofs = n_old_global_dofs - n_global_absorbed * T.block_size

    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    new_n_global_dofs = T.dim()
    new_ises = T._ises

    # Check that the FS data structures are rebuilt appropriately to match the new VOM topology
    fs_layout_matches_vom_topology_post_rebuild = (
        vom.cell_set.size == expected_n_local_nodes
        and T.dof_dset.size == expected_n_local_nodes
        and T.node_set.size == expected_n_local_nodes
        and T.node_set.total_size == expected_n_local_nodes
        and T.node_count == expected_n_local_nodes
        and T.dof_count == expected_n_local_nodes * T.block_size
        and T.cell_node_list.shape == (vom.cell_set.size, 1)
        and T.global_numbering.getStorageSize() == expected_n_local_nodes
        and T.dm is T.dof_dset.dm
        and new_ises is T.dof_dset.field_ises
    )

    parallel_assert(
        fs_layout_matches_vom_topology_post_rebuild,
        "The FunctionSpace layout does not match the rebuilt VOM topology",
    )

    assert new_n_global_dofs == expected_n_global_dofs


@pytest.mark.parallel([1, 3])
def test_vom_fs_rebuilds_under_successive_vom_rebuilds(vom, vom_fs):
    T = vom_fs.topological
    mutator = VertexOnlyMeshMutator(vom)

    block_size = T.block_size  # number of scalar DoFs per node
    v0 = vom.topology._topology_version

    shared_data_v0 = T._shared_data
    dof_dset_v0 = T.dof_dset
    dm_v0 = T.dm

    n_local_nodes_v0 = T.node_set.size
    n_global_dofs_v0 = T.dim()

    # First rebuild
    absorbed_v1 = np.arange(1, n_local_nodes_v0, 2, dtype=int)
    n_global_absorbed_v1 = vom.comm.allreduce(
        len(absorbed_v1),
        op=MPI.SUM,
    )

    expected_local_nodes_v1 = n_local_nodes_v0 - len(absorbed_v1)
    expected_global_dofs_v1 = n_global_dofs_v0 - n_global_absorbed_v1 * block_size

    mutator.rebuild_vom(absorbed_vom_indices=absorbed_v1)

    # Access thee FS state after the first rebuild
    v1 = vom.topology._topology_version
    shared_data_v1 = T._shared_data
    dof_dset_v1 = T.dof_dset
    dm_v1 = T.dm
    n_global_dofs_v1 = T.dim()

    fs_layout_matches_vom_topology_post_rebuild_1 = (
        v1 == v0 + 1
        and shared_data_v1 is not shared_data_v0
        and dof_dset_v1 is not dof_dset_v0
        and dm_v1 is not dm_v0
        and vom.cell_set.size == expected_local_nodes_v1
        and T.node_set.size == expected_local_nodes_v1
        and T.node_set.total_size == expected_local_nodes_v1
        and T.node_count == expected_local_nodes_v1
        and T.dof_count == expected_local_nodes_v1 * block_size
        and T.cell_node_list.shape == (vom.cell_set.size, 1)
        and n_global_dofs_v1 == expected_global_dofs_v1
    )

    parallel_assert(
        fs_layout_matches_vom_topology_post_rebuild_1,
        "The FunctionSpace was not rebuilt correctly after the first VOM mutation"
    )

    # Second rebuild
    absorbed_v2 = np.arange(1, vom.cell_set.size, 2, dtype=int)
    n_global_absorbed_v2 = vom.comm.allreduce(
        len(absorbed_v2),
        op=MPI.SUM,
    )

    expected_local_nodes_v2 = expected_local_nodes_v1 - len(absorbed_v2)
    expected_global_dofs_v2 = (
        expected_global_dofs_v1 - n_global_absorbed_v2 * block_size
    )

    mutator.rebuild_vom(absorbed_vom_indices=absorbed_v2)

    # Access the FS state after the second rebuild
    v2 = vom.topology._topology_version
    shared_data_2 = T._shared_data
    dof_dset_2 = T.dof_dset
    dm_2 = T.dm
    n_global_dofs_2 = T.dim()

    fs_layout_matches_vom_topology_post_rebuild_2 = (
        v2 == v1 + 1
        and shared_data_2 is not shared_data_v1
        and dof_dset_2 is not dof_dset_v1
        and dm_2 is not dm_v1
        and vom.cell_set.size == expected_local_nodes_v2
        and T.node_set.size == expected_local_nodes_v2
        and T.node_set.total_size == expected_local_nodes_v2
        and T.node_count == expected_local_nodes_v2
        and T.dof_count == expected_local_nodes_v2 * block_size
        and T.cell_node_list.shape == (vom.cell_set.size, 1)
        and n_global_dofs_2 == expected_global_dofs_v2
    )

    parallel_assert(
        fs_layout_matches_vom_topology_post_rebuild_2,
        "The FunctionSpace was not rebuilt correctly after the second VOM mutation",
    )


@pytest.mark.parallel([1, 3])
def test_vom_fs_rebuils_to_matche_empty_vom(vom, vom_fs):
    T = vom_fs.topological

    # First ensure all topology-dependent structures exist and have been computed
    _ = T.dof_dset
    _ = T.global_numbering
    _ = T.dm
    _ = T._ises
    _ = T.cell_node_list
    _ = T.cell_node_map()

    # Mutate the VOM: remove all points from every rank
    mutator = VertexOnlyMeshMutator(vom)
    old_local_count = vom.cell_set.size
    absorbed_local_indices = np.arange(old_local_count, dtype=int)
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    # Access the new state post rebuild
    node_set = T.node_set
    dof_dset = T.dof_dset
    layout_vec = dof_dset.layout_vec
    global_numbering = T.global_numbering
    dm = T.dm
    ises = T._ises
    cell_node_list = T.cell_node_list
    cell_node_map = T.cell_node_map()
    global_dim = T.dim()

    # Check that all FS data structures are now of size 0
    fs_layout_is_empty = (
        vom.cell_set.size == 0
        and vom.cell_set.total_size == 0
        and node_set.size == 0
        and node_set.total_size == 0
        and dof_dset.size == 0
        and dof_dset.total_size == 0
        and T.node_count == 0
        and T.dof_count == 0
        and global_dim == 0
        and layout_vec.getLocalSize() == 0
        and layout_vec.getSize() == 0
        and global_numbering.getStorageSize() == 0
        and cell_node_list.shape == (0, 1)
        and cell_node_list.size == 0
        and cell_node_map.values.size == 0
        and cell_node_map.values_with_halo.size == 0
        and dm is dof_dset.dm
        and all(iset.getSize() == 0 for iset in ises)
    )

    parallel_assert(
        fs_layout_is_empty,
        "The FunctionSpace data structures are non-empty despite the VOM being empty"
    )


@pytest.mark.parallel(nprocs=3)
def test_vom_fs_rebuilds_under_parallel_migration(parent_mesh, vom, vom_fs):
    mutator = VertexOnlyMeshMutator(vom)
    T = vom_fs.topological

    old_global_dim = T.dim()
    old_local_nodes = T.node_set.size
    old_dof_dset = T.dof_dset
    old_ises = T._ises
    old_dm = T.dm

    # Migrate the first particle on rank 0 to a neighbouring rank
    midpoints = cell_midpoints(parent_mesh, with_halos=True)

    new_coords = vom.coordinates.dat.data_ro.copy()
    destination_rank = None

    ghost_cell_ids = np.arange(parent_mesh.cell_set.size, parent_mesh.cell_set.total_size, dtype=int)

    if parent_mesh.comm.rank == 0:
        # New point is the midpoint of the first ghost cell
        target_cell = ghost_cell_ids[0]
        new_coords[0] = midpoints[target_cell]

        _, sf_leaves, sf_remotes = parent_mesh.topology_dm.getPointSF().getGraph()

        leaf_owning_ranks = {
            leaf: remote[0]
            for leaf, remote in zip(sf_leaves, sf_remotes)
        }

        target_plex_cell = parent_mesh.topology.cell_closure[target_cell, -1]  # Firedrake cell ID -> plex cell ID
        destination_rank = leaf_owning_ranks.get(target_plex_cell)

    destination_rank = parent_mesh.comm.bcast(destination_rank, root=0)

    parallel_assert(destination_rank is not None and destination_rank != 0, "Rank 0 failed to choose a destination rank to migrate its particle to")

    # Commit new state on every rank
    new_parent_cells, new_ref_coords = locate_points(parent_mesh, new_coords)
    vom.coordinates.dat.data_wo[:] = new_coords
    mutator.commit_reference_state(new_parent_cells, new_ref_coords)

    mutator.rebuild_vom()

    expected_local_nodes = old_local_nodes

    if parent_mesh.comm.rank == 0:
        expected_local_nodes -= 1

    if parent_mesh.comm.rank == destination_rank:
        expected_local_nodes += 1

    # Check that the FS data structures are rebuilt appropriately to match the new VOM topology
    fs_layout_matches_migration = (
        vom.cell_set.size == expected_local_nodes
        and T.node_set.size == expected_local_nodes
        and T.node_set.total_size == expected_local_nodes  # no ghost points by construction
        and T.dof_dset.size == expected_local_nodes
        and T.node_count == expected_local_nodes
        and T.dof_count == expected_local_nodes * T.block_size
        and T.cell_node_list.shape == (expected_local_nodes, 1)
        and T.global_numbering.getStorageSize() == expected_local_nodes
        and T.dof_dset.layout_vec.getLocalSize() == expected_local_nodes * T.block_size
        and T.dof_dset is not old_dof_dset
        and T.dm is not old_dm
        and T._ises is not old_ises
        and T._ises is T.dof_dset.field_ises
        and T.dim() == old_global_dim  # no points absorbed
    )

    parallel_assert(fs_layout_matches_migration, "The FunctionSpace layout does not match the migrated VOM topology")
