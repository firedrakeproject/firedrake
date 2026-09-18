# cython: language_level=3

# Low-level numbering for multigrid support
import cython
import numpy as np
from firedrake.cython import dmcommon
from firedrake.petsc import PETSc
from firedrake.utils import IntType
from pyop2.mpi import MPI

cimport numpy as np
cimport petsc4py.PETSc as PETSc
np.import_array()

include "petschdr.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
def get_entity_renumbering(PETSc.DM plex, PETSc.Section section, entity_type):
    """
    Given a section numbering a type of topological entity, return the
    renumberings from original plex numbers to new firedrake numbers
    (and vice versa)

    :arg plex: The DMPlex object
    :arg section: The Section defining the renumbering
    :arg entity_type: The type of entity (either ``"cell"`` or
        ``"vertex"``)
    """
    cdef:
        PetscInt start, end, p, ndof, entity
        np.ndarray old_to_new
        np.ndarray new_to_old

    if entity_type == "cell":
        start, end = plex.getHeightStratum(0)
    elif entity_type == "vertex":
        start, end = plex.getDepthStratum(0)
    else:
        raise RuntimeError("Entity renumbering for entities of type %s not implemented",
                           entity_type)

    old_to_new = np.empty(end - start, dtype=PETSc.IntType)
    new_to_old = np.empty(end - start, dtype=PETSc.IntType)

    for p in range(start, end):
        CHKERR(PetscSectionGetDof(section.sec, p, &ndof))
        if ndof > 0:
            CHKERR(PetscSectionGetOffset(section.sec, p, &entity))
            new_to_old[entity] = p - start
            old_to_new[p - start] = entity

    return old_to_new, new_to_old


@cython.boundscheck(False)
@cython.wraparound(False)
def coarse_to_fine_nodes(Vc, Vf, const PetscInt[:, ::1] coarse_to_fine_cells):
    cdef:
        const PetscInt[:, ::1] fine_map, coarse_map
        PetscInt[:, ::1] coarse_to_fine_map
        const PetscInt[::1] coarse_offset, fine_offset
        const PetscInt[::1] coarse_offset_quotient, fine_offset_quotient
        PetscInt i, j, k, ll, m, node, fine, layer
        PetscInt coarse_node_layer, fine_layer, fine_node_layer
        PetscInt coarse_per_cell, fine_per_cell, fine_cell_per_coarse_cell, coarse_cells
        PetscInt fine_layers, coarse_layer, coarse_layers, ratio
        bint extruded

    fine_map = Vf.cell_node_map().values
    coarse_map = Vc.cell_node_map().values

    fine_cell_per_coarse_cell = coarse_to_fine_cells.shape[1]
    extruded = Vc.extruded
    coarse_cells = coarse_map.shape[0]
    coarse_per_cell = coarse_map.shape[1]
    fine_per_cell = fine_map.shape[1]

    if extruded:
        coarse_offset = Vc.offset
        fine_offset = Vf.offset
        coarse_layers = Vc.mesh().layers - 1
        fine_layers = Vf.mesh().layers - 1

        ratio = fine_layers // coarse_layers
        assert ratio * coarse_layers == fine_layers  # check ratio is an int
        coarse_offset_quotient = np.zeros(coarse_per_cell, dtype=IntType)
        fine_offset_quotient = np.zeros(fine_per_cell, dtype=IntType)
        if Vc.offset_quotient is not None:
            coarse_offset_quotient = Vc.offset_quotient
        if Vf.offset_quotient is not None:
            fine_offset_quotient = Vf.offset_quotient

    ndof = fine_per_cell * fine_cell_per_coarse_cell
    if extruded:
        ndof *= ratio
    coarse_to_fine_map = np.full((Vc.dof_dset.total_size,
                                  ndof),
                                 -1,
                                 dtype=IntType)
    for i in range(coarse_cells):
        for j in range(coarse_per_cell):
            node = coarse_map[i, j]
            if extruded:
                for coarse_layer in range(coarse_layers):
                    k = 0
                    for ll in range(fine_cell_per_coarse_cell):
                        fine = coarse_to_fine_cells[i, ll]
                        if fine < 0:
                            k += fine_per_cell * ratio
                            continue
                        for layer in range(ratio):
                            fine_layer = coarse_layer * ratio + layer
                            coarse_node_layer = (coarse_layer + coarse_offset_quotient[j]) % coarse_layers
                            coarse_node_layer -= coarse_offset_quotient[j] % coarse_layers
                            for m in range(fine_per_cell):
                                fine_node_layer = (fine_layer + fine_offset_quotient[m]) % fine_layers
                                fine_node_layer -= fine_offset_quotient[m] % fine_layers
                                coarse_to_fine_map[node + coarse_offset[j]*coarse_node_layer, k] = (fine_map[fine, m] +
                                                                                                    fine_offset[m]*fine_node_layer)
                                k += 1
            else:
                k = 0
                for ll in range(fine_cell_per_coarse_cell):
                    fine = coarse_to_fine_cells[i, ll]
                    if fine < 0:
                        k += fine_per_cell
                        continue
                    for m in range(fine_per_cell):
                        coarse_to_fine_map[node, k] = fine_map[fine, m]
                        k += 1

    return np.asarray(coarse_to_fine_map)


@cython.boundscheck(False)
@cython.wraparound(False)
def fine_to_coarse_nodes(Vf, Vc, const PetscInt[:, ::1] fine_to_coarse_cells):
    cdef:
        const PetscInt[:, ::1] fine_map, coarse_map
        PetscInt[:, ::1] fine_to_coarse_map
        const PetscInt[::1] coarse_offset, fine_offset
        const PetscInt[::1] coarse_offset_quotient, fine_offset_quotient
        PetscInt i, j, k, ll, node, fine_layer, fine_layers, coarse_layer, coarse_layers, ratio
        PetscInt fine_node_layer, coarse_node_layer
        PetscInt coarse_per_cell, fine_per_cell, coarse_cell, fine_cells
        bint extruded

    fine_map = Vf.cell_node_map().values
    coarse_map = Vc.cell_node_map().values

    extruded = Vc.extruded
    coarse_per_cell = coarse_map.shape[1]
    fine_per_cell = fine_map.shape[1]

    if extruded:
        coarse_offset = Vc.offset
        fine_offset = Vf.offset
        coarse_layers = Vc.mesh().layers - 1
        fine_layers = Vf.mesh().layers - 1

        ratio = fine_layers // coarse_layers
        assert ratio * coarse_layers == fine_layers  # check ratio is an int
        coarse_offset_quotient = np.zeros(coarse_per_cell, dtype=IntType)
        fine_offset_quotient = np.zeros(fine_per_cell, dtype=IntType)
        if Vc.offset_quotient is not None:
            coarse_offset_quotient = Vc.offset_quotient
        if Vf.offset_quotient is not None:
            fine_offset_quotient = Vf.offset_quotient

    fine_cells = fine_to_coarse_cells.shape[0]
    coarse_per_fine = fine_to_coarse_cells.shape[1]
    fine_to_coarse_map = np.full((Vf.dof_dset.total_size,
                                  coarse_per_fine*coarse_per_cell),
                                 -1,
                                 dtype=IntType)

    for i in range(fine_cells):
        for ll, coarse_cell in enumerate(fine_to_coarse_cells[i, :]):
            if coarse_cell < 0:
                continue
            for j in range(fine_per_cell):
                node = fine_map[i, j]
                if extruded:
                    for fine_layer in range(fine_layers):
                        coarse_layer = fine_layer // ratio
                        fine_node_layer = (fine_layer + fine_offset_quotient[j]) % fine_layers
                        fine_node_layer -= fine_offset_quotient[j] % fine_layers
                        for k in range(coarse_per_cell):
                            coarse_node_layer = (coarse_layer + coarse_offset_quotient[k]) % coarse_layers
                            coarse_node_layer -= coarse_offset_quotient[k] % coarse_layers
                            fine_to_coarse_map[node + fine_offset[j]*fine_node_layer, k] = (
                                coarse_map[coarse_cell, k] + coarse_offset[k]*coarse_node_layer)
                else:
                    for k in range(coarse_per_cell):
                        fine_to_coarse_map[node, coarse_per_cell*ll + k] = coarse_map[coarse_cell, k]

    return np.asarray(fine_to_coarse_map)


def create_lgmap(PETSc.DM dm):
    """Create a local to global map for all points in the given DM.

    :arg dm: The DM to create the map for.

    Returns a petsc4py LGMap."""
    cdef:
        PETSc.IS iset = PETSc.IS()
        PETSc.LGMap lgmap = PETSc.LGMap()
        PetscInt *indices
        PetscInt i, size

    # Not necessary on one process
    if dm.comm.size == 1:
        return None
    CHKERR(DMPlexCreatePointNumbering(dm.dm, &iset.iset))
    CHKERR(ISLocalToGlobalMappingCreateIS(iset.iset, &lgmap.lgm))
    CHKERR(ISLocalToGlobalMappingGetSize(lgmap.lgm, &size))
    CHKERR(ISLocalToGlobalMappingGetBlockIndices(lgmap.lgm, <const PetscInt**>&indices))
    for i in range(size):
        if indices[i] < 0:
            indices[i] = -(indices[i]+1)

    CHKERR(ISLocalToGlobalMappingRestoreBlockIndices(lgmap.lgm, <const PetscInt**>&indices))

    return lgmap


cdef PetscInt num_owned_cells(PETSc.DM dm) except? -1:
    """Number of cells this rank owns, i.e. the number of Firedrake cell
    numbers the DM's cell numbering hands out to non-ghost cells.

    Parameters
    ----------
    dm : PETSc.DM
        The DMPlex encapsulating the mesh topology, with its PyOP2 entity
        classes already marked.

    Returns
    -------
    PetscInt
        The number of core plus owned cells.

    """
    return dmcommon.get_entity_classes(dm)[dm.getDimension(), 1]


@cython.boundscheck(False)
@cython.wraparound(False)
def transform_source_points(PETSc.DM dm):
    """Find the point that produced each point of a transformed DMPlex.

    Parameters
    ----------
    dm : PETSc.DM
        A DMPlex made by a transform, such as a refinement, of a DMPlex on
        which ``setSaveTransform`` was called first.

    Returns
    -------
    numpy.ndarray
        For each point of ``dm``, the point of the original DMPlex that
        produced it.

    """
    cdef:
        PETSc.PetscDMPlexTransform transform = NULL
        PetscInt pStart, pEnd, p, source
        PetscInt[::1] points

    CHKERR(DMPlexGetTransform(dm.dm, &transform))
    if transform == NULL:
        raise ValueError(
            "The DMPlex did not save its transform; call setSaveTransform "
            "before creating it so hierarchy point maps can be built"
        )
    pStart, pEnd = dm.getChart()
    points = np.empty(pEnd - pStart, dtype=IntType)
    for p in range(pStart, pEnd):
        CHKERR(DMPlexTransformGetSourcePoint(transform, p, NULL, NULL, &source, NULL))
        points[p - pStart] = source
    return np.asarray(points)


def compose_points(outer, inner):
    """Compose two point maps.

    Parameters
    ----------
    outer : numpy.ndarray
        The map to apply second.
    inner : numpy.ndarray
        The map to apply first, with -1 where it has no point.

    Returns
    -------
    numpy.ndarray
        ``outer[inner]``, with -1 where ``inner`` has no point.

    """
    points = np.full(inner.shape, -1, dtype=IntType)
    found = inner >= 0
    points[found] = outer[inner[found]]
    return points


def overlapped_fine_to_coarse_points(coarse_mesh, fine_mesh, fine_to_coarse_points,
                                     coarse_lgmap, fine_lgmap):
    """Renumber a refinement of unoverlapped DMPlexes onto the DMPlexes of two meshes.

    A hierarchy refines unoverlapped DMPlexes and adds overlap only when it
    builds each mesh. The saved transform relates the points of the
    unoverlapped DMPlexes, so this function carries both ends of the relation
    over to the meshes through the global point numbers.

    Parameters
    ----------
    coarse_mesh, fine_mesh : MeshGeometry
        The coarse mesh, and the mesh built from the refinement of its
        unoverlapped DMPlex.
    fine_to_coarse_points : numpy.ndarray
        For each point of the unoverlapped fine DMPlex, the point of the
        unoverlapped coarse DMPlex that produced it, as given by
        `transform_source_points`.
    coarse_lgmap, fine_lgmap : PETSc.LGMap or None
        The point local-to-global maps of the unoverlapped coarse and fine
        DMPlexes, as given by `create_lgmap`. These maps are ``None`` on a
        serial communicator, where the overlapped and unoverlapped point
        numberings are identical.

    Returns
    -------
    numpy.ndarray
        For each point of ``fine_mesh.topology_dm``, the point of
        ``coarse_mesh.topology_dm`` that it was refined from, or -1 for a
        point that only the overlap has.

    """
    if coarse_mesh.comm.size == 1:
        # On one process there is no overlap, so the numberings agree.
        return fine_to_coarse_points
    pStart, pEnd = fine_mesh.topology_dm.getChart()
    points = np.arange(pStart, pEnd, dtype=IntType)
    create_lgmap(fine_mesh.topology_dm).apply(points, result=points)
    points = fine_lgmap.applyInverse(points, PETSc.LGMap.MapMode.MASK)
    points = compose_points(fine_to_coarse_points, points)
    coarse_lgmap.apply(points, result=points)
    return create_lgmap(coarse_mesh.topology_dm).applyInverse(points, PETSc.LGMap.MapMode.MASK)


def coarse_to_fine_cells(coarse_mesh, fine_mesh, fine_to_coarse_points):
    """Build the cell maps between two meshes from the refinement of their points.

    Parameters
    ----------
    coarse_mesh, fine_mesh : MeshGeometry
        The coarse and fine meshes.
    fine_to_coarse_points : numpy.ndarray
        For each point of ``fine_mesh.topology_dm``, the point of
        ``coarse_mesh.topology_dm`` that it was refined from, or -1.

    Returns
    -------
    coarse_to_fine : numpy.ndarray
        For each owned coarse cell, the owned fine cells refined from it, in
        increasing order. Every row is as wide as the busiest coarse cell on
        any process, so a coarse cell with fewer fine cells has its row
        right-padded with -1.
    fine_to_coarse : numpy.ndarray
        A column with the owned coarse cell that each owned fine cell was
        refined from, or -1 where there is none.

    """
    ncoarse = coarse_mesh.cell_set.size
    nfine = fine_mesh.cell_set.size
    cStart, cEnd = coarse_mesh.topology_dm.getHeightStratum(0)
    fStart, _ = fine_mesh.topology_dm.getHeightStratum(0)
    coarse_cells, _ = get_entity_renumbering(coarse_mesh.topology_dm, coarse_mesh._cell_numbering, "cell")
    _, fine_points = get_entity_renumbering(fine_mesh.topology_dm, fine_mesh._cell_numbering, "cell")

    parents = fine_to_coarse_points[fine_points[:nfine] + fStart]
    # A submesh cell can be refined from a point that is not a coarse cell.
    is_cell = (cStart <= parents) & (parents < cEnd)
    parents[is_cell] = coarse_cells[parents[is_cell] - cStart]
    parents[~is_cell | (parents >= ncoarse)] = -1

    fine = np.flatnonzero(parents >= 0).astype(IntType)
    coarse = parents[fine]
    order = np.argsort(coarse, kind="stable")
    counts = np.bincount(coarse, minlength=ncoarse)
    width = coarse_mesh.comm.allreduce(int(counts.max(initial=0)), op=MPI.MAX)
    coarse_to_fine = np.full((ncoarse, width), -1, dtype=IntType)
    columns = np.arange(len(order)) - np.repeat(np.cumsum(counts) - counts, counts)
    coarse_to_fine[coarse[order], columns] = fine[order]
    return coarse_to_fine, parents.reshape(-1, 1)


@cython.boundscheck(False)
@cython.wraparound(False)
def filter_labels(PETSc.DM dm, keep, *label_names):
    """Remove labels from points that are not in keep.
    :arg dm: DM object with labels.
    :arg keep: subsection of the DMs chart on which to retain label values.
    :arg label_names: names of labels (strings) to clear.
    When refining, every point "underneath" the refined entity
    receives its label. But we typically have labels applied only to
    entities of a given stratum height (and rely on that elsewhere),
    so clear the labels from everything else.
    """
    cdef:
        PetscInt pStart, pEnd, kStart, kEnd, p, value
        DMLabel dmlabel = NULL

    pStart, pEnd = dm.getChart()
    kStart, kEnd = keep

    for label in label_names:
        if not dm.hasLabel(label):
            # Nothing to clear here.
            continue
        label = label.encode()
        CHKERR(DMGetLabel(dm.dm, <const char*>label, &dmlabel))
        for p in range(pStart, pEnd):
            if p < kStart or p >= kEnd:
                CHKERR(DMLabelGetValue(dmlabel, p, &value))
                if value >= 0:
                    CHKERR(DMLabelClearValue(dmlabel, p, value))
