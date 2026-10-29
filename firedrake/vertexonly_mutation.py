import numpy as np
from numpy.typing import ArrayLike
from firedrake.utils import IntType
from firedrake.mesh import MeshGeometry, VertexOnlyMeshTopology
from firedrake.function import _migrate_dg0_coef
from firedrake.petsc import PETSc
import firedrake.cython.dmcommon as dmcommon
from mpi4py import MPI

class VertexOnlyMeshMutator:
    """Mutate a vertex-only mesh in place."""
    def __init__(self, vom: MeshGeometry) -> None:
        self.vom = vom
        self.parent_mesh = self.vom._parent_mesh

        if not isinstance(self.vom.topology, VertexOnlyMeshTopology):
            raise TypeError(
            "The VertexOnlyMeshMutator expects a vertex-only mesh object, with a VertexOnlyMeshTopology."
        )

        if self.parent_mesh.extruded:
            raise NotImplementedError(
                "The VertexOnlyMeshMutator does not yet support vertex-only meshes immersed in extruded parent meshs."
            )

    def commit_reference_state(self, new_parent_cells: np.ndarray, new_refcoords: np.ndarray) -> None:
        """Commit updated parent-cell and reference-coordinate data.

        Parameters
        ----------
        new_parent_cells : numpy.ndarray
            Parent-cell numbers for locally owned VOM points. The expected
            shape is ``(n_owned,)``, matching the VertexOnlyMeshTopology's ``cell_parent_cell_list field``. 
            Values are assumed to be in the current VOM ordering.
        new_refcoords : numpy.ndarray
            Reference coordinates for locally owned VOM points. The expected
            shape is ``(n_owned, parent_tdim)``. Values are assumed to be in the current VOM ordering.
        """
        swarm = self.vom.topology_dm
        topology = self.vom.topology

        n_owned = topology.cells.owned.local_size
        vom_to_swarm = topology._new_to_old_point_renumbering[:n_owned]  # VOM cell ID -> swarm point ID

        new_parent_cells = np.asarray(new_parent_cells, dtype=IntType)

        assert new_parent_cells.shape == (n_owned,), "Expected the new parent cell list to be a one dimensional array"

        new_refcoords = np.asarray(new_refcoords, dtype=float)

        cell_id_name = swarm.getCellDMActive().getCellID()
        with (
            swarm.field("parentcellnum") as parentcellnum_field,
            swarm.field(cell_id_name) as swarm_parentcellnum_field,
            swarm.field("refcoord") as refcoord_field,
        ):
            parentcellnum_field[vom_to_swarm, 0] = new_parent_cells
            plex_ids = self.parent_mesh.topology._new_to_old_cell_numbering[new_parent_cells]
            swarm_parentcellnum_field[vom_to_swarm, 0] = plex_ids
            refcoord_field[vom_to_swarm, :] = new_refcoords

        topology._parent_cell_assignment_version += 1

        # Write the update reference coordinates
        # NOTE: Since new_ref_coords is already in VOM ordering and assuming the VOM has not been reodered when this function is called,
        # we can safely write into the reference coordinates array.
        self.vom.reference_coordinates.dat.data_wo[:] = new_refcoords

    def rebuild_vom(self, absorbed_vom_indices: ArrayLike | None = None) -> None:
        """Rebuild the vertex-only mesh (VOM) from its currently stored state.
        
        The VOM must contain the target coordinates, reference coordinates,
        and parent-cell numbers before this method is called. The rebuild
        redistributes the remaining particles and refreshes topology-dependent data.
    
        Parameters
        ----------
        absorbed_vom_indices : array_like, optional
            One-dimensional array containing the local VOM indices of
            particles to remove. Indices are interpreted in the current
            local VOM ordering. By default, no particles are removed.
    
        Notes
        -----
        This is a collective operation over the parent mesh communicator.
        Every rank must call it, although each rank may provide different
        local particle indices.
    
        The method modifies the VOM, and its underlying DMSwarm, in place. 
        It rebuilds the topology numbering, particle IDs, coordinate fields, 
        clears topology-dependent caches, and increments the topology version.
        """
        old_local_size = self._migrate_swarm(absorbed_vom_indices)
        self._rebuild_topology()
        step_sf = self._build_topology_step_sf(old_local_size)
        self._refresh_topology_fields()


    def _migrate_swarm(self, absorbed_vom_indices: ArrayLike | None = None) -> int:
        """Remove selected points and migrate the resulting swarm.

        Parameters
        ----------
        absorbed_vom_indices
            Local indices of VOM points to be removed.

        Returns
        -------
        int
            Number of owned VOM points before migration.
        """
        comm = self.parent_mesh.comm
        swarm = self.vom.topology_dm
        topology = self.vom.topology
        
        n_owned = topology.cells.owned.local_size
    
        # Read data from the current VOM
        coords_local = self.vom.coordinates.dat.data_ro.copy()
        refcoords_local = self.vom.reference_coordinates.dat.data_ro.copy()
        parent_cell_nums_local = topology.cell_parent_cell_list[:n_owned].copy()

        parent_plex = self.parent_mesh.topology_dm
        parent_topology = self.parent_mesh.topology
        plex_parent_cells_local = parent_topology._new_to_old_cell_numbering[parent_cell_nums_local]

        # Migrate each particle to the rank that owns its parent cell.
        owner_ranks_local = np.full(n_owned, comm.rank, dtype=IntType)
        owner_plex_parent_cells_local = plex_parent_cells_local.copy()
        if comm.size > 1:
            # The pointSF of the parent DMPlex is not set up in serial
            # PointSF example:
            # ----------------
            # On rank 0:
            # leaf 37  <-  (1, 12) -> plex point 37 on rank 0 is a copy of rank 1's plex point 12
            # leaf 41  <-  (1, 15) -> plex point 41 on rank 0 is a copy of rank 1's plex point 15
            # ...
            nroots, ilocal, iremote = parent_plex.getPointSF().getGraph()
            point_owner_ranks = np.full(parent_topology.num_points, comm.rank, dtype=IntType)
            point_owner_indices = np.arange(parent_topology.num_points, dtype=IntType)

            point_owner_ranks[ilocal] = iremote[:, 0]
            point_owner_indices[ilocal] = iremote[:, 1]

            owner_ranks_local = point_owner_ranks[plex_parent_cells_local]
            owner_plex_parent_cells_local = point_owner_indices[plex_parent_cells_local]

        # The global index of each particle in the current version.
        offset = comm.exscan(n_owned) or 0  # offset for this rank in global arrays
        global_idxs_local = np.arange(offset, offset + n_owned, dtype=IntType)  # indices for this rank's owned particles in global arrays

        # Each particle keeps its position in the (original) input-ordering VOM through every rebuild
        # Read these positions in the current VOM order
        vom_to_swarm = topology._new_to_old_point_renumbering[:n_owned]
        with swarm.field("inputrank") as input_ranks, swarm.field("inputindex") as input_indices:
            input_ranks_local = input_ranks.ravel()[vom_to_swarm].copy()
            input_indices_local = input_indices.ravel()[vom_to_swarm].copy()
    
        # Remove particles from the VOM
        absorbed = (
            np.empty(0, dtype=IntType)
            if absorbed_vom_indices is None
            else np.asarray(absorbed_vom_indices, dtype=IntType)
        )
    
        is_one_dim_local = absorbed.ndim == 1
        is_one_dim_global = comm.allreduce(
            is_one_dim_local,
            op=MPI.LAND,
        )
    
        if not is_one_dim_global:
            raise ValueError(
                "`absorbed_vom_indices` must be a one-dimensional array on every rank."
            )
    
        # Do a rank-local check: are all indices of particles to be removed valid?
        in_range_local = np.all((absorbed >= 0) & (absorbed < n_owned))
    
        # Check for collective agreeement: does the above check hold on every rank?
        in_range_global = comm.allreduce(in_range_local, op=MPI.LAND)
        if not in_range_global:
            raise ValueError("`absorbed_vom_indices` contains an invalid local particle index on at least one rank.")
    
        is_absorbed = np.zeros(n_owned, dtype=bool)
        is_absorbed[absorbed] = True
        survivors = ~is_absorbed
        n_survivors_local = int(survivors.sum())  # total number of survivors on this rank
    
        # Mutate the DMSwarm in-place
        # First compact the swarm to the new particles count
        swarm.setLocalSizes(n_survivors_local, 0)

        # Rewrite the swarm's fields
        # NOTE: The result is that the mutated swarm's point order now becomes the current VOM order
        cell_id_name = swarm.getCellDMActive().getCellID() # returns the name of the swarm field that stores each particle's parent cell number (in the old ordering)
        swarm.set_field("DMSwarmPIC_coor", coords_local[survivors])
        swarm.set_field(cell_id_name, owner_plex_parent_cells_local[survivors])
        swarm.set_field("globalindex", global_idxs_local[survivors])
        swarm.set_field("refcoord", refcoords_local[survivors])
        swarm.set_field("DMSwarm_rank", owner_ranks_local[survivors])
        swarm.set_field("inputrank", input_ranks_local[survivors])
        swarm.set_field("inputindex", input_indices_local[survivors])

        # Redistribute particles accross ranks
        swarm.migrate(remove_sent_points=True)
    
        # Convert the received plex cell numbers to Firedrake cell numbers on this rank.
        with swarm.field(cell_id_name) as swarm_plex_parent_cells:
            plex_parent_cells = swarm_plex_parent_cells.ravel().copy()
            swarm.set_field("parentcellnum", parent_topology._old_to_new_cell_numbering_is.indices[plex_parent_cells])

        # The migration sends every particle to its owning rank, so the swarm has no halo points.
        empty = np.empty(0, dtype=IntType)
        swarm.set_halo_sf(swarm.getLocalSize(), empty, empty)

        return n_owned

    def _rebuild_topology(self) -> None:
        """Rebuild the point numbering of the VOM topology after swarm migration."""
        comm = self.parent_mesh.comm
        topology = self.vom.topology
        topology._recompute_point_numbering()

        # The migration sends every particle to its owning rank (rank that owns its parent cell)
        # so we check that no rank has ghost particles after the revuild
        no_ghosts_local = topology.cells.owned.local_size == topology.num_cells()
        no_ghosts_global = comm.allreduce(no_ghosts_local, op=MPI.LAND)

        if not no_ghosts_global:
            raise AssertionError("Unexpected VOM ghost points post migration")

    def _build_topology_step_sf(self, old_local_size: int,) -> PETSc.SF:
        """Create and record the SF for the latest topology change.

        Parameters
        ----------
        old_local_size
            Number of local particles before swarm migration.

        Returns
        -------
        PETSc.SF
            SF with the cells of the rebuilt topology as leaves and the cells
            of the previous topology as roots.
        """
        comm = self.parent_mesh.comm
        swarm = self.vom.topology_dm
        topology = self.vom.topology

        # `_migrate_swarm` stores the global index of each particle in the previous version
        # in the `globalindex` field. The owned cells of each rank are numbered contiguously,
        # so the first global index on each rank gives the rank and the local index of each particle.
        old_offsets = np.cumsum([0, *comm.allgather(old_local_size)[:-1]], dtype=IntType) # every rank knows every rank's offset
        with swarm.field("globalindex") as global_indices:
            old_global_idxs = global_indices.ravel()
            old_ranks = np.searchsorted(old_offsets, old_global_idxs, side="right") - 1
            remote = np.empty(2 * len(old_global_idxs), dtype=IntType)
            remote[0::2] = old_ranks # old rank
            remote[1::2] = old_global_idxs - old_offsets[old_ranks] # old local index = global index - rank offset

            step_sf = PETSc.SF().create(comm=comm)
            step_sf.setGraph(old_local_size, topology._old_to_new_point_renumbering, remote)

            # The SF stored under key k maps VOM version k-1 (roots) to VOM version k (leaves).

        topology._topology_step_sfs[topology._topology_version] = step_sf
        return step_sf

    def _refresh_topology_fields(self) -> None:
        """Move the fields of the VOM to the rebuilt topology."""
        fields = [self.vom.coordinates.topological, self.vom._particle_ids]
        if self.vom.reference_coordinates is not None:
            # A VOM's reference coordinates remain undefined when building a VOM embedded
            # in a parent mesh with topological dimension 0.
            fields.append(self.vom.reference_coordinates)
        for cfunc in fields:
            cfunc._migrate_to_current_topology()
