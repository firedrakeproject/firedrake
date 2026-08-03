import numpy as np
from numpy.typing import ArrayLike
from firedrake.utils import IntType
from firedrake.mesh import MeshGeometry, VertexOnlyMeshTopology
from firedrake.function import migrate_dg0_dat
import firedrake.cython.dmcommon as dmcommon
from pyop2.mpi import MPI


class VertexOnlyMeshMutator:
    """Mutate a VertexOnlyMesh in place as particles move through its parent mesh."""
    def __init__(self, vom: MeshGeometry) -> None:
        self.vom = vom
        self.parent_mesh = self.vom._parent_mesh

        if not isinstance(self.vom.topology, VertexOnlyMeshTopology):
            raise TypeError(
            "The VertexOnlyMeshMutator requires a mesh with a VertexOnlyMeshTopology."
        )

        if self.parent_mesh.extruded:
            raise NotImplementedError(
                "The VertexOnlyMeshMutator does not yet support VertexOnlyMeshes immersed in an extruded parent mesh."
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

        Notes
        -----
        The parent-cell numbers and reference coordinates must be
        geometrically consistent: each row of ``new_refcoords`` must locate
        the corresponding point in the cell specified by
        ``new_parent_cells``.

        DMSwarm fields use DMSwarm ordering, whereas the inputs to this
        method use the current VOM ordering.
        """
        swarm = self.vom.topology_dm
        topology = self.vom.topology

        n_owned = self.vom.cell_set.size
        vom_to_swarm = self.vom.cell_closure[:n_owned, -1]  # VOM cell ID -> swarm point ID

        new_parent_cells = np.asarray(new_parent_cells, dtype=IntType)

        assert new_parent_cells.shape == (n_owned,), "Expected the new parent cell list to be supplied as a one dimensional array"

        new_refcoords = np.asarray(new_refcoords, dtype=float)

        cell_id_name = swarm.getCellDMActive().getCellID()
        with (
            swarm.field("parentcellnum") as parentcellnum_field,
            swarm.field(cell_id_name) as swarm_parentcellnum_field,
            swarm.field("refcoord") as refcoord_field,
        ):
            parentcellnum_field[vom_to_swarm, 0] = new_parent_cells
            plex_ids = self.parent_mesh.topology.cell_closure[new_parent_cells, -1].reshape((-1, 1))
            swarm_parentcellnum_field[vom_to_swarm, :] = plex_ids
            refcoord_field[vom_to_swarm, :] = new_refcoords

        # Invalidate cached properties that depend on parent cell ownership
        for name in (
            "cell_parent_base_cell_list",
            "cell_parent_base_cell_map",
            "cell_parent_extrusion_height_list",
            "cell_parent_extrusion_height_map",
        ):
            if name in topology.__dict__:
                del topology.__dict__[name]

        # Mutate the parent cell list
        if "cell_parent_cell_list" in topology.__dict__:
            topology.__dict__["cell_parent_cell_list"][:n_owned] = new_parent_cells

        # Write the update reference coordinates
        # NOTE: Since new_ref_coords is already in VOM ordering and assuming the VOM has not been reodered when this function is called,
        # we can safely write into the reference coordinates array.
        ref_coords_func = self.vom.reference_coordinates
        ref_coords_func.dat.data[:] = new_refcoords


    def rebuild_vom(self, absorbed_vom_indices: ArrayLike | None = None) -> None:
        """Rebuild the vertex-only mesh (VOM) from its currently stored state.

        The VOM must contain the desired coordinates, reference coordinates,
        and parent-cell numbers before this method is called. The rebuild
        removes selected particles, redistributes the remaining particles,
        and refreshes topology-dependent data.

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
        and topology-dependent caches, and increments the topology version.
        """
        comm = self.parent_mesh.comm
        swarm = self.vom.topology_dm
        topology = self.vom.topology
        gdim = self.vom.geometric_dimension

        parent_plex = self.parent_mesh.topology_dm
        parent_tdim = self.parent_mesh.topological_dimension

        n_local = topology.cell_set.size  # number of particles owned by this rank

        # Read data from the current VOM
        coords_local = self.vom.coordinates.dat.data_ro.copy()
        refcoords_local = self.vom.reference_coordinates.dat.data_ro.copy()
        parent_cell_nums_local = topology.cell_parent_cell_list[:n_local].copy()

        # Hand over particles in ghost cells to the owning ranks at using the parent mesh DMPlex's pointSF
        # PointSF example:
        # ----------------
        # On rank 0:
        # leaf 37  <-  (1, 12) -> plex point 37 on rank 0 is a copy of rank 1's plex point 12
        # leaf 41  <-  (1, 15) -> plex point 41 on rank 0 is a copy of rank 1's plex point 15
        # ...

        owned_ranks_local = np.full(n_local, comm.rank)

        plex_parent_cell_nums_local = self.parent_mesh.topology.cell_closure[parent_cell_nums_local, -1]  # map Firedrake parent cell number -> local DMPlex point number
        new_plex_parent_cell_nums_local = plex_parent_cell_nums_local.copy()

        # redistribution happens in parallel only
        if comm.size > 1:
            nroots, ilocal, iremote = parent_plex.getPointSF().getGraph()
            owning_ranks = dict(zip(ilocal, iremote[:, 0]))  # dict {leaf idx: owning rank number}
            local_idxs_on_owning_ranks = dict(zip(ilocal, iremote[:, 1]))  # dict {leaf idx: local idx on owning rank}

            n_cells_local = self.parent_mesh.cell_set.size  # number of parent cells owned by this rank
            ghost_parent_cells = parent_cell_nums_local >= n_cells_local

            for i in np.where(ghost_parent_cells)[0]:
                owned_ranks_local[i] = owning_ranks[plex_parent_cell_nums_local[i]]
                new_plex_parent_cell_nums_local[i] = local_idxs_on_owning_ranks[plex_parent_cell_nums_local[i]]

        offset = comm.scan(n_local, op=MPI.SUM) - n_local  # offset for this rank in global arrays
        global_idxs_local = np.arange(offset, offset + n_local, dtype=IntType)  # indices for this rank's owned particles in global arrays
        input_idxs_local = np.arange(n_local, dtype=IntType)  # local indices for this rank's owned particles in local arrays

        input_ranks_local = np.full(n_local, comm.rank)  # input ranks for this rank's owned particles
        # NOTE: we assume that the current VOM contains only particles owned by this rank
        # after each rebuild, ``swarm.migrate(remove_sent_points=True)`` hands particles to their
        # new owning ranks and removes sent copies, and the VOM is asserted to have no ghost particles.

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

        # Rank-local check: are all indices of particles to be removed valid?
        in_range_local = np.all((absorbed >= 0) & (absorbed < n_local)) if absorbed.size else True

        # Collective agreement: does the above check hold on every rank?
        in_range_global = comm.allreduce(in_range_local, op=MPI.LAND)
        if not in_range_global:
            raise ValueError("`absorbed_vom_indices` contains an invalid local particle index on at least one rank.")

        is_absorbed = np.zeros(n_local, dtype=bool)
        is_absorbed[absorbed] = True

        survivors = ~is_absorbed
        n_survivors_local = int(survivors.sum())  # total number of survivors on this rank

        # Reset the IO Swarm
        # NOTE: it seems costly to re-build a new IO swarm every time the VOM gets rebuild.
        # Should we instead maintain in some way a reference to the very first IO swarm?
        topology.input_ordering_swarm = None

        # Mutate the DMSwarm of the VOM in-place
        # First compact the swarm to the new particles count
        swarm.setLocalSizes(n_survivors_local, 0)

        # Now write the output arrays (filtered to include the survivor particles only) into the compacted swarm arrays
        # NOTE: The result is that the swarm's point order now becomes the current local VOM order
        cell_id_name = swarm.getCellDMActive().getCellID()
        with (
            swarm.field("DMSwarmPIC_coor") as coord_field,
            swarm.field(cell_id_name) as swarm_parentcellnum_field,
            swarm.field("globalindex") as globalindex_field,
            swarm.field("refcoord") as refcoord_field,
            swarm.field("DMSwarm_rank") as rank_field,
            swarm.field("inputrank") as inputrank_field,
            swarm.field("inputindex") as inputindex_field,
        ):
            coord_field[...] = coords_local[survivors]
            swarm_parentcellnum_field[:, 0] = new_plex_parent_cell_nums_local[survivors]
            globalindex_field[:, 0] = global_idxs_local[survivors]
            refcoord_field[...] = refcoords_local[survivors]
            rank_field[:, 0] = owned_ranks_local[survivors]
            inputrank_field[:, 0] = input_ranks_local[survivors]
            inputindex_field[:, 0] = input_idxs_local[survivors]

        # Redistribute particles accross ranks
        swarm.migrate(remove_sent_points=True)

        # Reconstruct Firedrake cell numbers from the receiving rank's updated plex data
        with (
            swarm.field(cell_id_name) as swarm_parentcellnum_field,
            swarm.field("parentcellnum") as parentcellnum_field,
        ):
            swarm_plex_ids = swarm_parentcellnum_field[:, 0]

            parent_cells_plex_ids = self.parent_mesh.topology.cell_closure[:, -1]
            plex_to_parent_cells = {int(plex_id): parent_cell for parent_cell, plex_id in enumerate(parent_cells_plex_ids)}

            new_parent_cells = np.asarray(
                [
                    plex_to_parent_cells.get(int(plex_id))
                    for plex_id in swarm_plex_ids
                ],
                dtype=IntType
            )
            parentcellnum_field[:, 0] = new_parent_cells

        # Calling migrate above changed the chart so we reset the DMSwarm's pointSF
        sf = swarm.getPointSF()
        sf.setGraph(swarm.getLocalSize(), None, [])
        swarm.setPointSF(sf)

        # Rebuild the entity renumbering
        # `_renumber_entities` reads the swarm's cellid field (plex parent cell num) and for each swarm point sorts by its parent cell's Firedrake rank in the parent mesh
        # returns a PETSc IS mapping Firedrake cell j to swarm point _dm_renumbering[j]
        topology._dm_renumbering = topology._renumber_entities(reorder=True)

        # Clear stale entity class labels before re-marking
        for _label_name in ("pyop2_core", "pyop2_owned", "pyop2_ghost"):
            if swarm.hasLabel(_label_name):
                swarm.clearLabelStratum(_label_name, 1)

        # Mark all entities with their new classes before calling create_section which uses the DMSwarm's entity class labels
        dmcommon.mark_entity_classes_using_cell_dm(swarm)  # rewrites class labels on based on the ownership of the plex parent cell
        topology._entity_classes = dmcommon.get_entity_classes(swarm)  # reads swarm labels and store counts on the vom's topology

        # Rebuild _cell_numbering and _vertex_numbering from the new _dm_renumbering
        entity_dofs = np.array([1], dtype=IntType)  # 1 DoF per point

        # cell_numbering and vertex_numbering are PETSc ISes - translation tables between PETSc numbering of mesh entities and that of Firedrake's
        # For each plex point, they store two integers: the dof count and an offset (which happens to be the Firedrake's number of that plex point)
        topology._cell_numbering, _ = topology.create_section(entity_dofs)
        topology._vertex_numbering = topology._cell_numbering  # holds for VOM only

        # Invalidate cached topology-dependent properties
        self._invalidate_topology_properties()

        # NOTE: Every particle's DMSwarm_rank is an owner due to the handover,
        # `remove_sent_points` sends the particle to that rank and deletes the local copy
        # check here after caches has been cleared so `cell_set` is appropriately recomputed
        no_ghosts_local = self.vom.cell_set.size == self.vom.num_vertices()
        no_ghosts_global = comm.allreduce(no_ghosts_local, op=MPI.LAND)

        if not no_ghosts_global:
            raise AssertionError("Unexpected VOM ghost points post migration")

        # Build the one step SF corresponding to the current rebuild
        e_p_map = topology.cell_closure[:, -1]  # maps new VOM point number -> raw DMSwarm point
        ilocal = np.empty_like(e_p_map)  # inverts the `e_p_map`
        if len(e_p_map):
            cStart = e_p_map.min()
            ilocal[e_p_map - cStart] = np.arange(len(e_p_map))

        step_sf = VertexOnlyMeshTopology._make_input_ordering_sf(swarm, n_local, ilocal)

        # Increment the VOM topology version
        topology._topology_version = comm.allreduce(topology._topology_version + 1, op=MPI.MAX)

        # Store the one step SF under the new topology version number
        # maps VOM version k (new) -> VOM version k-1 (old) stored under key k
        topology._topology_step_sfs[topology._topology_version] = step_sf

        # Clear the shared FunctionSpace caches on the VOM
        topology._shared_data_cache.clear()

        # Migrate the particle ID field through the step SF
        topology._particle_ids = migrate_dg0_dat(topology._particle_ids, topology._particle_ids.function_space(), step_sf)

        # Rebuild coordinate fields
        coords_fs = self.vom._coordinates.function_space()
        ref_coords_fs = self.vom.reference_coordinates.function_space().topological

        coords_data = dmcommon.reordered_coords(swarm, coords_fs.dm.getDefaultSection(),
                                                (topology.num_vertices(), gdim))

        # Resize the CoordinatelessFunction dat buffer and assign new values
        self.vom._coordinates.dat = coords_fs.make_dat(val=coords_data, name=self.vom._coordinates.name())  # returns a new op2.Dat

        if parent_tdim > 0:
            ref_coords_data = dmcommon.reordered_coords(swarm, ref_coords_fs.dm.getDefaultSection(),
                                                        (topology.num_vertices(), parent_tdim), reference_coord=True)
            self.vom.reference_coordinates._data.dat = ref_coords_fs.make_dat(val=ref_coords_data, name=self.vom.reference_coordinates.name())  # returns a new op2.Dat
        else:
            # This should have been already set to None when the VOM was first constructed
            self.vom.reference_coordinates = None

    def _invalidate_topology_properties(self):
        # Delete cached properties so they get recomputed on next access using the updated swarm fields
        topology = self.vom.topology
        for name in (
            "exterior_facets",
            "interior_facets",
            "cell_to_facets",
            "cell_closure",
            "cell_set",
            "cell_parent_cell_list",
            "cell_parent_cell_map",
            "cell_parent_base_cell_list",
            "cell_parent_base_cell_map",
            "cell_parent_extrusion_height_list",
            "cell_parent_extrusion_height_map",
            "cell_global_index",
            "input_ordering",
            "input_ordering_sf",
            "input_ordering_without_halos_sf",
        ):
            if name in topology.__dict__:
                del topology.__dict__[name]
