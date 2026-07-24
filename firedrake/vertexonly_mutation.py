import numpy as np
from firedrake.petsc import PETSc

class EmptyVOMError(Exception):
    """Raised when all particles have been absorbed and the VertexOnlyMesh is empty."""
    pass

class VertexOnlyMeshUpdater:
    def __init__(self, vom):
        self.vom = vom
        self.parent_mesh = self.vom._parent_mesh

        if self.parent_mesh.extruded:
            raise NotImplementedError(
                "VertexOnlyMeshUpdater does not yet support VertexOnlyMeshes with an extruded parent mesh."
            )


    def commit_reference_state(self, next_parent_cells, new_refcoords):
        """Perform a reference-only VOM update.

        Updates the parent cell ownership and reference coordinates under the assumption that the tuple
        (next_parent_cells, new_refcoords) are geometrically consistent that is, ref_coords correctly represents 
        each point's reference coordinates in its new parent cell.

        NOTE: Swarm fields are in DMSwarm ordering but `next_parent_cells` and `new_refcoords` are in VOM ordering.
        """
        swarm = self.vom.topology_dm
        n_owned = self.vom.cell_set.size
        vom_to_swarm = self.vom.cell_closure[:n_owned, -1] # VOM cell ID -> DMSwarm point ID

        next_parent_cells = np.asarray(next_parent_cells, dtype=int).reshape((-1, 1))
        new_refcoords = np.asarray(new_refcoords, dtype=float)

        # Update Firedrake parent cell numbers
        arr = swarm.getField("parentcellnum")
        arr[vom_to_swarm, 0] = next_parent_cells[:, 0]
        swarm.restoreField("parentcellnum")

        # Update plex parent cell IDs
        cell_id_name = swarm.getCellDMActive().getCellID()
        arr = swarm.getField(cell_id_name)
        plex_ids = self.parent_mesh.topology.cell_closure[
            next_parent_cells.reshape(-1), -1
        ].reshape((-1, 1))

        arr[vom_to_swarm, :] = plex_ids
        swarm.restoreField(cell_id_name)

        # Update reference coordinates
        arr = swarm.getField("refcoord")
        arr[vom_to_swarm, :] = new_refcoords
        swarm.restoreField("refcoord")
        
        # 4) Invalidate cached topology
        # NOTE: Invalidating caches causes them to be recomputed on next access,
        # but we haven't updated all the fields at this stage yet.
        # Instead of invalidating all properties by calling `self.invalidate_topology_properties()`
        # we only delete the properties that depend on parent cell ownership.
        topology = self.vom.topology
        for name in (
            # "cell_parent_cell_list",
            # "cell_parent_cell_map",
            "cell_parent_base_cell_list",
            "cell_parent_base_cell_map",
            "cell_parent_extrusion_height_list",
            "cell_parent_extrusion_height_map",
        ):
            if name in topology.__dict__:
                del topology.__dict__[name]

            if name in self.vom.__dict__:
                del self.vom.__dict__[name]
        
        # Mutate the parent cell list instead of invalidating it
        if "cell_parent_cell_list" in topology.__dict__:
            topology.__dict__["cell_parent_cell_list"][:n_owned] = next_parent_cells[:]

        # 5) Update reference coordinates Function
        # NOTE: Since new_ref_coords is already in VOM ordering, AND assuming the VOM has not been reodered when this function is called,
        # we can safely modify the reference coordinates array.
        ref_coords_func = self.vom.reference_coordinates
        ref_coords_func.dat.data[:] = new_refcoords


    # NOTE: By the time rebuild_vom is called in the loop, the VOM already holds the post-step state
    # commit_reference_state has already written the new ref. coords and updated the parent cells in the VOM
    def rebuild_vom(self, absorbed_vom_indices=None):
        from firedrake.mesh import VertexOnlyMeshTopology
        from firedrake.function import migrate_dg0_dat
        from firedrake.utils import IntType
        import firedrake.cython.dmcommon as dmcommon
        from pyop2.mpi import MPI
        
        comm = self.parent_mesh.comm
        swarm = self.vom.topology_dm
        topology = self.vom.topology
        
        gdim = self.vom.geometric_dimension
        
        parent_plex = self.parent_mesh.topology_dm
        parent_tdim = self.parent_mesh.topological_dimension
        
        n_local = topology.cell_set.size # number of particles owned by this rank 

        # Read data from the current VOM
        coords_local = self.vom.coordinates.dat.data_ro.copy()
        reference_coords_local = self.vom.reference_coordinates.dat.data_ro.copy()
        parent_cell_nums_local = topology.cell_parent_cell_list[:n_local].ravel().copy()

        # TODO: Should we migrate all particles at rebuild time or only the ones that hit the partition boundary?
        # Hand over particles in ghost cells to the owning ranks (parallel-only)
        """
        rank 0's plex point SF:
        leaf 37  <-  (1, 12) means "my point 37 is a copy of rank 1's point 12"
        leaf 41  <-  (1, 15)
        ...
        """
        owned_ranks_local = np.full(n_local, comm.rank)

        parent_local_plex = self.parent_mesh.topology.cell_closure[parent_cell_nums_local, -1]
        plex_parent_cell_nums_local = parent_local_plex.copy()

        if comm.size > 1:
            nroots, ilocal, iremote = parent_plex.getPointSF().getGraph()
            owning_ranks = dict(zip(ilocal, iremote[:, 0])) # {leaf idx: owning rank number}
            local_idxs_on_owning_ranks = dict(zip(ilocal, iremote[:, 1])) # {leaf idx: local idx on owning rank}

            n_cells_local = self.parent_mesh.cell_set.size # number of parent cells owned by this rank
            ghost_parent_cells = parent_cell_nums_local.ravel() >= n_cells_local

            for i in np.where(ghost_parent_cells)[0]:
                owned_ranks_local[i] = owning_ranks[parent_local_plex[i]]
                plex_parent_cell_nums_local[i] = local_idxs_on_owning_ranks[parent_local_plex[i]]

        offset = comm.scan(n_local, op=MPI.SUM) - n_local # offset for this rank in the global array
        global_idxs_local = np.arange(offset, offset + n_local, dtype=IntType) # indices for this rank in the global array
        input_idxs_local = np.arange(n_local, dtype=IntType)
        input_ranks_local = np.full(n_local, comm.rank)

        # Absorb particles
        absorbed = np.array([] if absorbed_vom_indices is None else absorbed_vom_indices, dtype=int)
        
        # do a rank-local check: are all indices of particles to be absorbed valid?
        in_range_local = np.all((absorbed >= 0) & (absorbed < n_local)) if len(absorbed) else True

        # collective agreement: is this valid on every rank?
        in_range_global = comm.allreduce(in_range_local, op=MPI.LAND)
        if not in_range_global:
            raise ValueError("absorbed VOM index is out of range on some rank")

        is_absorbed = np.zeros(n_local, dtype=bool)
        is_absorbed[absorbed] = True

        # Check that some particles are still alive
        survivors = ~is_absorbed 
        n_survivors_local = int(survivors.sum()) # survivors on this rank
        n_survivors_global = comm.allreduce(n_survivors_local, op=MPI.SUM)
        if n_survivors_global == 0:
            raise EmptyVOMError("All particles have left the domain (no points remaining in the VertexOnlyMesh).")

        # NOTE: How do we maintain a ref. to the very first IO VOM?
        topology.input_ordering_swarm = None

        # Mutate the current swarm in-place
        # Compact the swarm to n_survivors_local points
        swarm.setLocalSizes(n_survivors_local, 0)

        # Now write the output arrays (filtered to include survivors only) into the compacted swarm
        # The result is that the swarm's order now becomes the current VOM cell orderf
        swarm_coords = swarm.getField("DMSwarmPIC_coor").reshape((n_survivors_local, gdim))
        swarm_coords[...] = coords_local[survivors]
        swarm.restoreField("DMSwarmPIC_coor")

        cid = swarm.getCellDMActive().getCellID()
        swarm_parent_cell_nums = swarm.getField(cid).ravel()
        swarm_parent_cell_nums[...] = plex_parent_cell_nums_local[survivors]
        swarm.restoreField(cid)

        field_global_index = swarm.getField("globalindex").ravel()
        field_global_index[...] = global_idxs_local[survivors]
        swarm.restoreField("globalindex")

        field_reference_coords = swarm.getField("refcoord").reshape((n_survivors_local, parent_tdim))
        field_reference_coords[...] = reference_coords_local[survivors]
        swarm.restoreField("refcoord")

        field_parent_cell_nums = swarm.getField("parentcellnum").ravel()
        field_parent_cell_nums[...] = parent_cell_nums_local[survivors]
        swarm.restoreField("parentcellnum")

        field_rank = swarm.getField("DMSwarm_rank").ravel()
        field_rank[...] = owned_ranks_local[survivors]
        swarm.restoreField("DMSwarm_rank")

        field_input_rank = swarm.getField("inputrank").ravel()
        field_input_rank[...] = input_ranks_local[survivors]
        swarm.restoreField("inputrank")

        field_input_index = swarm.getField("inputindex").ravel()
        field_input_index[...] = input_idxs_local[survivors]
        swarm.restoreField("inputindex")

        # TODO:
        # When parent mesh is extruded, compute `base_parent_cell_nums`, `extrusion_heights`
        # and `plex_parent_cell_nums` based on `base_parent_cell_nums`
        
        # if self.parent_mesh.extruded:
        #     field_base_parent_cell_nums = swarm.getField("parentcellbasenum").ravel()
        #     field_extrusion_heights = swarm.getField("parentcellextrusionheight").ravel()
        #     field_base_parent_cell_nums[...] = base_parent_cell_nums[visible]
        #     field_extrusion_heights[...] = extrusion_heights[visible]
        #     swarm.restoreField("parentcellbasenum")
        #     swarm.restoreField("parentcellextrusionheight")
        
        # Redistribute particles accross ranks
        swarm.migrate(remove_sent_points=True)

        # Migrate changed the chart so we reset the swarm's pointSF
        sf = swarm.getPointSF()
        sf.setGraph(swarm.getLocalSize(), None, [])
        swarm.setPointSF(sf)

        # Rebuild entity renumbering on the existing topology
        # Reads the swarm's cellid field (plex parent cell num) and for each swarm point sorts by its parent cell's Firedrake rank in the parent mesh
        # Returns a PETSc IS mapping Firedrake cell j to swarm point perm[j]
        topology._dm_renumbering = topology._renumber_entities(reorder=True)

        # Clear stale entity class labels before re-marking
        for _label_name in ("pyop2_core", "pyop2_owned", "pyop2_ghost"):
            if swarm.hasLabel(_label_name):
                swarm.clearLabelStratum(_label_name, 1)
        
        
        # Mark all entities with their new classes before calling create_section which uses the DM's entity class labels
        dmcommon.mark_entity_classes_using_cell_dm(swarm) # rewrite the class labels on the swarm points based on the ownership of the plex parent cell
        topology._entity_classes = dmcommon.get_entity_classes(swarm) # read swarm labels and store counts on the vom's topology

        # Rebuild _cell_numbering and _vertex_numbering from the new _dm_renumbering
        entity_dofs = np.array([1], dtype=IntType)  # 1 DoF per point

        # cell_numbering and vertex_numbering are PETSc ISes - translation tables between PETSc numbering of mesh entities and that of Firedrake's
        # For each plex point, they store two integers: the dof count and an offset (which happens to be the Firedrake's number of that plex point)
        topology._cell_numbering, _ = topology.create_section(entity_dofs)
        topology._vertex_numbering = topology._cell_numbering # holds for VOM only

        # Build the step SF
        e_p_map = topology.cell_closure[:, -1] # new cell -> raw swarm point
        ilocal = np.empty_like(e_p_map)
        if len(e_p_map):
            cStart = e_p_map.min()
            ilocal[e_p_map - cStart] = np.arange(len(e_p_map))

        step_sf = VertexOnlyMeshTopology._make_input_ordering_sf(swarm, n_local, ilocal)

        # Increment the VOM topology version
        topology._topology_version = comm.allreduce(topology._topology_version + 1, op=MPI.MAX)

        # Store the one step SF under the new topology version number
        # maps VOM version k (new) -> VOM version k-1 (old) stored under key k
        topology._topology_step_sfs[topology._topology_version] = step_sf
        
        # Clear the shared FS caches on the VOM
        topology._shared_data_cache.clear()

        # Invalidate cached topological properties
        # Amongst other things, this triggers a recomputation of the IO SF on next access
        self._invalidate_topology_properties()
        
        # NOTE: Every particle's DMSwarm_rank is an owner due to the handover,
        # `remove_sent_points` sends the particle to that rank and deletes the local copy
        # (no rank keeps a ghost copy)
        assert self.vom.cell_set.size == self.vom.num_vertices(), "unexpected ghost particles"

        # Migrate the ID field through the step SF 
        topology._particle_ids = migrate_dg0_dat(topology._particle_ids, topology._particle_ids.function_space(), step_sf)

        # Rebuild coordinate fields
        coords_fs = self.vom._coordinates.function_space()
        ref_coords_fs = self.vom.reference_coordinates.function_space().topological

        coords_data = dmcommon.reordered_coords(swarm, coords_fs.dm.getDefaultSection(),
                                                (topology.num_vertices(), gdim))
        
        # Resize the CoordinatelessFunction dat buffer and assign new values
        self.vom._coordinates.dat = coords_fs.make_dat(val=coords_data, name=self.vom._coordinates.name()) # returns a new op2.Dat

        if parent_tdim > 0:
            ref_coords_data = dmcommon.reordered_coords(swarm, ref_coords_fs.dm.getDefaultSection(),
                                                        (topology.num_vertices(), parent_tdim), reference_coord=True)
            # Resize the CoordinatelessFunction dat buffer and assign new values
            # NOTE: Since reference_coordinates is a Function, ._data accesses the CoordinatelessFunction it wraps where the data buffer lives
            self.vom.reference_coordinates._data.dat = ref_coords_fs.make_dat(val=ref_coords_data, name=self.vom.reference_coordinates.name()) # returns a new op2.Dat
        else:
            # Should have been already set to None when the VOM was first constructed
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
            # Clear on the mesh topology object
            if name in topology.__dict__:
                del topology.__dict__[name]
