import firedrake.cython.dmcommon as dmcommon
from firedrake.cython.dmcommon import DistributedMeshOverlapType
from firedrake.halo import _get_mtype as get_mpi_type
from firedrake.mesh import MeshTopology
from pyop2.mpi import MPI


class RedistributedMeshTransfer:
    """Transfer data between a parent-owned mesh and its redistributed mesh."""

    def __init__(self, orig, redist, point_sf=None):
        self.orig = orig
        self.redist = redist
        self.point_sf = point_sf if point_sf is not None else redist.sfBC

    def _section_sf(self, root, leaf):
        point_sf = self.point_sf
        root_section = root.function_space().dm.getDefaultSection()
        leaf_section = leaf.function_space().dm.getDefaultSection()
        remote_offsets, _ = point_sf.distributeSection(root_section, leaf_section)
        return point_sf.createSectionSF(root_section, remote_offsets, leaf_section)

    def orig2redist(self, source, target):
        section_sf = self._section_sf(source, target)
        dtype, _ = get_mpi_type(source.dat)
        section_sf.bcastBegin(dtype,
                              source.dat.data_ro_with_halos,
                              target.dat.data_wo_with_halos,
                              MPI.REPLACE)
        section_sf.bcastEnd(dtype,
                            source.dat.data_ro_with_halos,
                            target.dat.data_wo_with_halos,
                            MPI.REPLACE)

    def redist2orig(self, source, target):
        section_sf = self._section_sf(target, source)
        dtype, _ = get_mpi_type(source.dat)
        section_sf.reduceBegin(dtype,
                               source.dat.data_ro_with_halos,
                               target.dat.data_wo_with_halos,
                               MPI.REPLACE)
        section_sf.reduceEnd(dtype,
                             source.dat.data_ro_with_halos,
                             target.dat.data_wo_with_halos,
                             MPI.REPLACE)


def distribute_overlap(dm, parameters):
    overlap_type, overlap = parameters.get(
        "overlap_type", (DistributedMeshOverlapType.FACET, 1)
    )
    if overlap_type == DistributedMeshOverlapType.NONE:
        if overlap > 0:
            raise ValueError("Cannot have NONE overlap with overlap > 0")
        return None
    elif overlap_type in [DistributedMeshOverlapType.FACET,
                          DistributedMeshOverlapType.RIDGE]:
        dmcommon.set_adjacency_callback(dm, overlap_type)
        sf = dm.distributeOverlap(overlap)
        dmcommon.clear_adjacency_callback(dm)
        return sf
    elif overlap_type == DistributedMeshOverlapType.VERTEX:
        return dm.distributeOverlap(overlap)
    else:
        raise ValueError("Unknown overlap type %r" % (overlap_type,))


def make_unoverlapped_dm(dm):
    """Effectively invert addOverlap().

    The resulting plex has the identical data structure as the one before
    addOverlap(). This is algorithmically guaranteed.
    """
    tdim = dm.getDimension()
    dm = dmcommon.submesh_create(dm, tdim, "depth", tdim, True)
    dm.removeLabel("pyop2_core")
    dm.removeLabel("pyop2_owned")
    dm.removeLabel("pyop2_ghost")
    dm.setRefinementUniform(True)
    return dm


def redistribute_dm(dm, parameters, grow_overlap=True):
    dm.removeLabel("pyop2_core")
    dm.removeLabel("pyop2_owned")
    dm.removeLabel("pyop2_ghost")

    distribute = parameters.get("partition", True)
    partitioner_type = parameters.get("partitioner_type")
    MeshTopology._set_partitioner(dm, distribute, partitioner_type)
    point_sf_orig = dm.distribute(overlap=0)
    if not grow_overlap:
        return point_sf_orig, point_sf_orig
    overlap_sf = distribute_overlap(dm, parameters)
    if overlap_sf is None:
        point_sf = point_sf_orig
    elif point_sf_orig is None:
        point_sf = overlap_sf
    else:
        point_sf = point_sf_orig.compose(overlap_sf)
    return point_sf_orig, point_sf


def dm_has_empty_rank(dm):
    cstart, cend = dm.getHeightStratum(0)
    return dm.comm.tompi4py().allreduce(cstart == cend, op=MPI.LOR)
