r"""This module implements mesh redistribution between MPI
communicators and data transfer between the two meshes.

The idea is that we may, perhaps due to refinement, wish to
redistribute our mesh to get better load balance. Often this might
occur in a multigrid hierarchy, where the coarsest mesh is not large
enough to distribute over all processes. In this situation, we would
like to be able to redistribute the refined mesh onto more processes,
but still be able to transfer data between levels.

The idea here is that we keep around the "before redistribution" mesh,
and provide facilities for transferring data (Functions) between the
two views of the same mesh.
"""
import numpy
import firedrake
from firedrake.halo import _get_mpi_type as numpy_to_mpi_type, _get_mtype as dat_to_mpi_type
from firedrake.petsc import PETSc
from firedrake.utils import cached_property
from mpi4py import MPI
from pyop2.datatypes import IntType


__all__ = ("RedistributedMeshManager", )


class RedistributedMeshManager(object):
    """
    Manage redistribution of a mesh onto a new communicator.

    :arg source_mesh: Mesh to redistribute (may be None on ranks that
        do not have it)
    :arg target_comm: Target communicator. All processes in
        source_mesh.comm must have a rank in target_comm.
    :arg distribution_parameters: Parameters to control the halo
        growth of the redistributed mesh.
    """
    def __init__(self, source_mesh, target_comm, distribution_parameters=None):
        self._sectionsf_cache = {}
        self.source_mesh = source_mesh
        if source_mesh is not None:
            overlapped = source_mesh._grown_halos
            dm = source_mesh._plex
            target_rank = target_comm.rank
        else:
            dm = None
            target_rank = -1
            overlapped = None
        # Everyone needs data that is only available on the ranks
        # where the source mesh is not None.
        self.bcast_rank = target_comm.allreduce(target_rank, op=MPI.MAX)
        self._source_already_overlapped = target_comm.bcast(overlapped, root=self.bcast_rank)
        migrationsf, distdm = PETSc.DMPlex.distributeToComm(dm, 0, target_comm)
        # These will be reconstructed.
        distdm.removeLabel("interior_facets")
        distdm.removeLabel("pyop2_core")
        distdm.removeLabel("pyop2_owned")
        distdm.removeLabel("pyop2_ghost")
        if distdm.handle == 0:
            # No redistribution occurred.
            assert migrationsf.handle == 0
            self.target_mesh = source_mesh
            self._source2target_migrationsf = None
        else:
            params = {}
            if distribution_parameters is not None:
                params.update(distribution_parameters)
            params["partition"] = False
            self.target_mesh = firedrake.Mesh(distdm, distribution_parameters=params)
            self._source2target_migrationsf = migrationsf

    @cached_property
    def migrationsf(self):
        """An SF, collective over :attr:`target_comm` that migrates
    points in the source mesh to points in the target mesh."""
        migrationsf = self._source2target_migrationsf
        if migrationsf is None:
            return migrationsf
        if not self._source_already_overlapped:
            # We redistributed before overlapping the source mesh
            if self.source_mesh is not None:
                self.source_mesh.init()
                overlapsf = self.source_mesh._overlapsf
                needs_rewrite = overlapsf is not None
            else:
                needs_rewrite = False

            needs_rewrite = self.target_mesh.comm.bcast(needs_rewrite, root=self.bcast_rank)

            if needs_rewrite:
                # Migration SF of redistributed DM references root
                # points in non-overlapped source DM. We have added an
                # overlap, so we need to rewrite those root points to
                # reference roots in the overlapped version of the
                # source DM. Do this by broadcasting the relabelled
                # roots over the migrationSF and rewriting the graph.
                if self.source_mesh is not None:
                    source_overlapsf = self.source_mesh._overlapsf
                    nroots, local, remote = source_overlapsf.getGraph()
                    relabelled_roots = numpy.full(nroots, -1, dtype=IntType)
                    indices, = numpy.where(remote[:, 0] == self.source_mesh.comm.rank)
                    relabelled_roots[remote[indices, 1]] = local[indices]
                    new_nroots = len(local)
                else:
                    relabelled_roots = numpy.empty(0, dtype=IntType)
                    new_nroots = 0
                _, mlocal, mremote = migrationsf.getGraph()
                new_roots = numpy.empty_like(mlocal)
                typ = numpy_to_mpi_type(IntType)
                migrationsf.bcastBegin(typ, relabelled_roots, new_roots)
                migrationsf.bcastEnd(typ, relabelled_roots, new_roots)
                mremote[:, 1] = new_roots[mlocal]
                migrationsf.setGraph(new_nroots, mlocal, mremote)

        # OK, now we have the correct migration sf onto the
        # non-overlapped target mesh, check if that also has an overlap.
        self.target_mesh.init()
        overlapsf = self.target_mesh._overlapsf
        if overlapsf is None:
            # No, migrationsf is fine.
            return migrationsf
        else:
            # Yes, must compose with the overlap migration
            return migrationsf.composeSF(overlapsf)

    def function_space_migration_sf(self, source, target):
        """Create an SF that migrates between the nodes of two function spaces.

        :arg source: The source function space (may be None), should
            be on same communicator as :attr:`source_mesh`.
        :arg target: The target function space, should be on same
            communicator as :attr:`target_mesh`.
        :returns: An SF that migrates nodes from the source to the target.

        The source and target function spaces must be the same, up to
        the mesh they are defined on. IOW, they must represent the
        same discretisation just with a different data layout.
        """
        if self.migrationsf is None:
            return None
        entity_dofs = target.finat_element.entity_dofs()
        nodes_per_entity = tuple(target.mesh().make_dofs_per_plex_entity(entity_dofs))
        try:
            return self._sectionsf_cache[nodes_per_entity]
        except KeyError:
            pass
        if source is None:
            source_section = None
        else:
            source_section = source.dm.getDefaultSection()
        target_section = target.dm.getDefaultSection()
        sf = PETSc.SF.createSectionMigrationSF(self.migrationsf, source_section, target_section)
        return self._sectionsf_cache.setdefault(nodes_per_entity, sf)

    def source_to_target(self, source, target):
        """Move data in a source function to a target function.

        :arg source: The source function (defined on
            :attr:`source_mesh`, may be None)
        :arg target: The target function (defined on
            :attr:`target_mesh`).

        The source and target function spaces must be the same, up to
        the mesh they are defined on. IOW, they must represent the
        same discretisation just with a different data layout.
        """
        if source is None:
            Vs = None
            rootdata = numpy.empty(0, dtype=target.dat.dtype)
        else:
            Vs = source.function_space()
            rootdata = source.dat.data_ro_with_halos
        sf = self.function_space_migration_sf(Vs, target.function_space())
        # FIXME: More exchanges than necessary
        leafdata = target.dat.data_with_halos
        if sf is None:
            leafdata[:] = rootdata[:]
        else:
            datatype, _ = dat_to_mpi_type(target.dat)
            sf.bcastBegin(datatype, rootdata, leafdata)
            sf.bcastEnd(datatype, rootdata, leafdata)

    def target_to_source(self, target, source):
        """Move data in a target function to a source function.

        :arg target: The target function (defined on
            :attr:`target_mesh`).
        :arg source: The source function (defined on
            :attr:`source_mesh`, may be None)

        The source and target function spaces must be the same, up to
        the mesh they are defined on. IOW, they must represent the
        same discretisation just with a different data layout.
        """
        if source is None:
            Vs = None
            rootdata = numpy.empty(0, dtype=target.dat.dtype)
        else:
            Vs = source.function_space()
            rootdata = source.dat.data_with_halos
        leafdata = target.dat.data_ro_with_halos
        sf = self.function_space_migration_sf(Vs, target.function_space())
        if sf is None:
            rootdata[:] = leafdata[:]
        else:
            datatype, _ = dat_to_mpi_type(target.dat)
            sf.reduceBegin(datatype, leafdata, rootdata, MPI.REPLACE)
            sf.reduceEnd(datatype, leafdata, rootdata, MPI.REPLACE)
