from firedrake import *
from firedrake.utils import IntType
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
    """Return the midpoint of each cell of the mesh, in the cell order of the mesh."""
    V = VectorFunctionSpace(mesh, "DG", 0)
    x = Function(V).interpolate(SpatialCoordinate(mesh))

    return (x.dat.data_ro_with_halos if with_halos else x.dat.data_ro).copy()


def locate_points(mesh, points):
    """Get local data (reference coordinates + parent cells) from physical point coordinates."""
    parent_cells, refcoords, _ = mesh.locate_cells_ref_coords_and_dists(points)
    return np.asarray(parent_cells, dtype=int), np.asarray(refcoords, dtype=float)

def assert_fs_layout(vom, T, n_expected):
    """Check that the layout of the topological FunctionSpace T matches ``n_expected`` owned VOM points on this rank."""
    bs = T.block_size
    n_ises = sum(iset.getLocalSize() for iset in T.field_ises)
    checks = {
        "owned VOM cells": vom.topology.cells.owned.local_size == n_expected,
        "all VOM cells": vom.topology.num_cells() == n_expected,
        "axes": T.axes.local_size == n_expected * bs,
        "dof_count": T.dof_count == n_expected * bs,
        "section": T.section.getStorageSize() == n_expected * bs,
        "template_vec": T.template_vec.getLocalSize() == n_expected * bs,
        "field_ises": n_ises == n_expected * bs,
        "cell_node_map_dat": T.cell_node_map_dat.data_ro.shape == (n_expected, 1),
    }
    for name, ok in checks.items():
        parallel_assert(ok, f"{name} does not match {n_expected} points")

@pytest.mark.parallel([1, 3])
def test_vom_fs_caches_recompute_after_vom_mutation(vom, vom_fs):
    """Check that FunctionSpace caches get invalidated and recomputed after a VOM topology change."""
    T = vom_fs.topological

    names = [
        "axes", "dm_axes", "nodal_layout_axes", "plex_axes", "nodal_axes", "_nodes_axis",
        "dm_axis_constraints", "nodal_axis_constraints", "section", "dm", "template_vec",
        "field_ises", "local_ises", "_lgmap", "cell_node_map", "cell_node_map_dat",
    ]
    old_vals = {name: getattr(T, name) for name in names}

    vom.topology._topology_version += 1

    for name in names:
        parallel_assert(getattr(T, name) is not old_vals[name], f"{name} was not recomputed")


@pytest.mark.parallel([1, 3])
def test_vom_fs_data_resizes_to_match_new_vom_topology(vom, vom_fs):
    T = vom_fs.topological
    bs = T.block_size # number of scalar DoFs per node
    n_owned = vom.topology.cells.owned.local_size
    n_old_global_dofs = T.dim() # total number of scalar DoFs

    # Remove every other point on each rank
    absorbed = np.arange(0, n_owned, 2, dtype=IntType)
    # rebuild VOM
    mutator = VertexOnlyMeshMutator(vom)
    mutator._migrate_swarm(absorbed)
    mutator._rebuild_topology()
    
    n_expected = n_owned - len(absorbed)
    n_global_absorbed = vom.comm.allreduce(len(absorbed), op=MPI.SUM)

    assert_fs_layout(vom, T, n_expected)
    assert T.dim() == n_old_global_dofs - n_global_absorbed * bs

    

# FIXME
@pytest.mark.skip(reason="Needs VertexOnlyMeshMutator._refresh_topology_fields")
@pytest.mark.parallel([1, 3])
def test_vom_fs_rebuilds_under_successive_vom_rebuilds(vom, vom_fs):
    """Check the FunctionSpace layout after two successive VOM rebuilds."""
    T = vom_fs.topological
    mutator = VertexOnlyMeshMutator(vom)
    bs = T.block_size

    n_expected = vom.topology.cells.owned.local_size
    n_expected_global_dofs = T.dim()
    for rebuild in (1, 2):
        version = vom.topology._topology_version
        old = {name: getattr(T, name) for name in ("axes", "section", "dm")}

        # Remove every other point on each rank, starting with the second point.
        absorbed = np.arange(1, n_expected, 2, dtype=IntType)
        mutator.rebuild_vom(absorbed_vom_indices=absorbed)
        n_expected -= len(absorbed)
        n_expected_global_dofs -= vom.comm.allreduce(len(absorbed), op=MPI.SUM) * bs

        parallel_assert(vom.topology._topology_version == version + 1)
        for name, value in old.items():
            parallel_assert(getattr(T, name) is not value, f"rebuild {rebuild}: {name} was not recomputed")
        assert_fs_layout(vom, T, n_expected)
        assert T.dim() == n_expected_global_dofs


@pytest.mark.parallel([1, 3])
def test_vom_fs_rebuilds_to_match_empty_vom(vom, vom_fs):
    """Check that the FunctionSpace layout is empty after every point is removed."""
    T = vom_fs.topological

    # Compute all topology-dependent structures before the rebuild
    for name in ("axes", "section", "dm", "field_ises", "cell_node_map", "cell_node_map_dat"):
        getattr(T, name)

    # Remove every point on every rank
    n_owned = vom.topology.cells.owned.local_size
    # rebuild VOM
    mutator = VertexOnlyMeshMutator(vom)
    mutator._migrate_swarm(np.arange(n_owned, dtype=IntType))
    mutator._rebuild_topology()

    assert_fs_layout(vom, T, 0)
    parallel_assert(T.template_vec.getSize() == 0, "the global template vector is not empty")
    assert T.dim() == 0


@pytest.mark.parallel(nprocs=3)
def test_vom_fs_rebuilds_under_parallel_migration(parent_mesh, vom, vom_fs):
    """Check the FunctionSpace layout after one point moves from rank 0 to another rank."""
    T = vom_fs.topological
    mutator = VertexOnlyMeshMutator(vom)
    parent_topology = parent_mesh.topology

    old_global_dim = T.dim()
    old_local_nodes = vom.topology.cells.owned.local_size
    old = {name: getattr(T, name) for name in ("axes", "section", "dm", "field_ises")}

    # Move the first point on rank 0 to the midpoint of its first ghost cell.
    # Firedrake numbers the owned cells first, so the ghost cells follow them.
    midpoints = cell_midpoints(parent_mesh, with_halos=True)
    new_coords = vom.coordinates.dat.data_ro.copy()
    destination_rank = None
    if parent_mesh.comm.rank == 0:
        target_cell = parent_topology.cells.owned.local_size
        new_coords[0] = midpoints[target_cell]

        _, sf_leaves, sf_remotes = parent_mesh.topology_dm.getPointSF().getGraph()
        target_plex_cell = parent_topology._new_to_old_cell_numbering[target_cell]
        (destination_rank,) = sf_remotes[sf_leaves == target_plex_cell, 0]
        destination_rank = int(destination_rank)
    destination_rank = parent_mesh.comm.bcast(destination_rank, root=0)
    parallel_assert(destination_rank != 0, "Rank 0 failed to choose a destination rank for its point")

    new_parent_cells, new_ref_coords = locate_points(parent_mesh, new_coords)
    vom.coordinates.dat.data_wo[:] = new_coords
    mutator.commit_reference_state(new_parent_cells, new_ref_coords)
    # rebuild VOM
    mutator._migrate_swarm()
    mutator._rebuild_topology()

    n_expected = old_local_nodes
    if parent_mesh.comm.rank == 0:
        n_expected -= 1
    if parent_mesh.comm.rank == destination_rank:
        n_expected += 1

    for name, value in old.items():
        parallel_assert(getattr(T, name) is not value, f"{name} was not recomputed")
    assert_fs_layout(vom, T, n_expected)
    assert T.dim() == old_global_dim