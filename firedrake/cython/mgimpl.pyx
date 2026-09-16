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
def coarse_to_fine_nodes(Vc, Vf, np.ndarray coarse_to_fine_cells):
    cdef:
        np.ndarray fine_map, coarse_map, coarse_to_fine_map
        np.ndarray coarse_offset, fine_offset
        PetscInt i, j, k, ll, m, node, fine, layer
        PetscInt coarse_per_cell, fine_per_cell, fine_cell_per_coarse_cell, coarse_cells
        PetscInt fine_layer, fine_layers, coarse_layer, coarse_layers, ratio
        bint extruded

    fine_map = Vf.cell_node_map().values
    coarse_map = Vc.cell_node_map().values

    fine_cell_per_coarse_cell = coarse_to_fine_cells.shape[1]
    extruded = Vc.extruded

    if extruded:
        coarse_offset = Vc.offset
        fine_offset = Vf.offset
        coarse_layers = Vc.mesh().layers - 1
        fine_layers = Vf.mesh().layers - 1

        ratio = fine_layers // coarse_layers
        assert ratio * coarse_layers == fine_layers  # check ratio is an int
    coarse_cells = coarse_map.shape[0]
    coarse_per_cell = coarse_map.shape[1]
    fine_per_cell = fine_map.shape[1]

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
                            for m in range(fine_per_cell):
                                coarse_to_fine_map[node + coarse_offset[j]*coarse_layer, k] = (fine_map[fine, m] +
                                                                                               fine_offset[m]*fine_layer)
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

    return coarse_to_fine_map


@cython.boundscheck(False)
@cython.wraparound(False)
def fine_to_coarse_nodes(Vf, Vc, np.ndarray fine_to_coarse_cells):
    cdef:
        np.ndarray fine_map, coarse_map, fine_to_coarse_map
        np.ndarray coarse_offset, fine_offset
        PetscInt i, j, k, ll, node, fine_layer, fine_layers, coarse_layer, coarse_layers, ratio
        PetscInt coarse_per_cell, fine_per_cell, coarse_cell, fine_cells
        bint extruded

    fine_map = Vf.cell_node_map().values
    coarse_map = Vc.cell_node_map().values

    extruded = Vc.extruded

    if extruded:
        coarse_offset = Vc.offset
        fine_offset = Vf.offset
        coarse_layers = Vc.mesh().layers - 1
        fine_layers = Vf.mesh().layers - 1

        ratio = fine_layers // coarse_layers
        assert ratio * coarse_layers == fine_layers  # check ratio is an int

    fine_cells = fine_to_coarse_cells.shape[0]
    coarse_per_fine = fine_to_coarse_cells.shape[1]
    coarse_per_cell = coarse_map.shape[1]
    fine_per_cell = fine_map.shape[1]
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
                        for k in range(coarse_per_cell):
                            fine_to_coarse_map[node + fine_offset[j]*fine_layer, k] = coarse_map[coarse_cell, k] + coarse_offset[k]*coarse_layer
                else:
                    for k in range(coarse_per_cell):
                        fine_to_coarse_map[node, coarse_per_cell*ll + k] = coarse_map[coarse_cell, k]

    return fine_to_coarse_map


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
def coarse_to_fine_cells(coarse_mesh, fine_mesh,
                         PETSc.DM coarse_dm, PETSc.DM fine_dm,
                         clgmaps=None, flgmaps=None):
    """Build cell maps from the transform that produced ``fine_dm``.

    Parameters
    ----------
    coarse_mesh, fine_mesh : MeshGeometry
        The coarse and fine Firedrake meshes. Their cell numberings define the
        numbering of the returned maps.
    coarse_dm, fine_dm : PETSc.DM
        The halo-free DMPlex pair on which the transform was applied.
    clgmaps, flgmaps : tuple or None
        The ``(halo-free, overlapped)`` point local-to-global maps for the
        coarse and fine meshes. Pass ``None`` when the Firedrake meshes use the
        same DMPlexes as the transform.

    Returns
    -------
    tuple of numpy.ndarray
        The ``(coarse_to_fine, fine_to_coarse)`` maps in Firedrake cell
        numbering. Rows in ``coarse_to_fine`` are padded with -1.

    """
    cdef:
        PetscInt cStart, cEnd, fStart, fEnd
        PetscInt ncoarse, nfine, c, fine_cell, coarse_cell
        PetscInt parent, max_children
        np.ndarray co2n, fn2o
        np.ndarray idx, found, permuted
        PetscInt[::1] child_counts
        PetscInt[:, ::1] coarse_to_fine
        PetscInt[:, ::1] fine_to_coarse
        PETSc.PetscDMPlexTransform transform = NULL

    ncoarse = coarse_mesh.cell_set.size
    nfine = fine_mesh.cell_set.size
    cStart, cEnd = coarse_dm.getHeightStratum(0)
    fStart, fEnd = fine_dm.getHeightStratum(0)
    co2n, _ = get_entity_renumbering(
        coarse_mesh.topology_dm, coarse_mesh._cell_numbering, "cell",
    )
    _, fn2o = get_entity_renumbering(
        fine_mesh.topology_dm, fine_mesh._cell_numbering, "cell",
    )

    if clgmaps is not None and clgmaps[0] is not None:
        cno, co = clgmaps
        fno, fo = flgmaps
        # Translate fine Firedrake cells from the overlapped DM to the
        # halo-free fine DM, where the transform relation is defined.
        fn2o = fn2o + fine_mesh.topology_dm.getHeightStratum(0)[0]
        fo.apply(fn2o, result=fn2o)
        fn2o = fno.applyInverse(fn2o, PETSc.LGMap.MapMode.MASK)

        # Translate the coarse point-to-cell numbering to the halo-free coarse
        # DM. The map changes both the point values and the array index, so
        # scatter it into the coarse cell stratum rather than slicing it.
        idx = np.arange(coarse_mesh.cell_set.total_size, dtype=PETSc.IntType)
        idx += coarse_mesh.topology_dm.getHeightStratum(0)[0]
        co.apply(idx, result=idx)
        idx = cno.applyInverse(idx, PETSc.LGMap.MapMode.MASK)
        found = idx >= 0
        permuted = np.full(cEnd - cStart, -1, dtype=PETSc.IntType)
        permuted[idx[found] - cStart] = co2n[found]
        co2n = permuted

    CHKERR(DMPlexGetTransform(fine_dm.dm, &transform))
    if transform == NULL:
        raise RuntimeError("The fine DMPlex did not retain its refinement transform")

    fine_to_coarse = np.full((nfine, 1), -1, dtype=IntType)
    child_counts = np.zeros(ncoarse, dtype=IntType)
    for c in range(nfine):
        fine_cell = fn2o[c]
        if not (fStart <= fine_cell < fEnd):
            continue
        CHKERR(DMPlexTransformGetSourcePoint(
            transform, fine_cell, NULL, NULL, &coarse_cell, NULL,
        ))
        if not (cStart <= coarse_cell < cEnd):
            continue
        parent = co2n[coarse_cell - cStart]
        if not (0 <= parent < ncoarse):
            continue
        fine_to_coarse[c, 0] = parent
        child_counts[parent] += 1

    max_children = 0
    for c in range(ncoarse):
        if child_counts[c] > max_children:
            max_children = child_counts[c]
    max_children = fine_dm.comm.tompi4py().allreduce(max_children, op=MPI.MAX)
    coarse_to_fine = np.full((ncoarse, max_children), -1, dtype=IntType)
    child_counts[:] = 0
    for c in range(nfine):
        parent = fine_to_coarse[c, 0]
        if parent >= 0:
            coarse_to_fine[parent, child_counts[parent]] = c
            child_counts[parent] += 1

    return np.asarray(coarse_to_fine), np.asarray(fine_to_coarse)


@cython.boundscheck(False)
@cython.wraparound(False)
def coarse_to_fine_submesh_cells(coarse_mesh, fine_mesh,
                                 PETSc.DM coarse_dm, PETSc.DM fine_dm,
                                 clgmaps, flgmaps):
    """Build cell maps for consecutive submeshes of a transformed hierarchy.

    The submesh cells are points in the parent DMPlex. The saved transform on
    the parent fine DMPlex therefore supplies their parent relation, even
    though the submesh DMPlexes did not arise from that transform directly.

    Parameters
    ----------
    coarse_mesh, fine_mesh : MeshGeometry
        The coarse and fine submeshes whose Firedrake cell numbering defines
        the returned maps.
    coarse_dm, fine_dm : PETSc.DM
        The halo-free parent DMPlex pair on which the transform was applied.
    clgmaps, flgmaps : tuple
        The parent mesh ``(halo-free, overlapped)`` point local-to-global maps.

    Returns
    -------
    tuple of numpy.ndarray
        The ``(coarse_to_fine, fine_to_coarse)`` maps in submesh Firedrake cell
        numbering. Rows in ``coarse_to_fine`` are padded with -1.

    """
    cdef:
        PetscInt ncoarse, nfine, c, parent
        PetscInt coarse_subStart, fine_subStart
        PetscInt coarse_parent_pStart, coarse_parent_pEnd
        PetscInt coarse_point, fine_point
        PetscInt max_children
        np.ndarray coarse_subcell_to_point, fine_subcell_to_point
        np.ndarray fine_dm_points
        np.ndarray valid, mapped_points
        PetscInt[::1] coarse_point_to_cell
        PetscInt[::1] fine_parent_points
        PetscInt[::1] coarse_dm_points
        PetscInt[::1] child_counts
        PetscInt[:, ::1] coarse_to_fine
        PetscInt[:, ::1] fine_to_coarse
        PETSc.IS coarse_subpoint_is, fine_subpoint_is
        const PetscInt *coarse_subpoints = NULL
        const PetscInt *fine_subpoints = NULL
        PETSc.PetscDMPlexTransform transform = NULL

    ncoarse = coarse_mesh.cell_set.size
    nfine = fine_mesh.cell_set.size
    coarse_subStart, _ = coarse_mesh.topology_dm.getHeightStratum(0)
    fine_subStart, _ = fine_mesh.topology_dm.getHeightStratum(0)
    coarse_parent_pStart, coarse_parent_pEnd = coarse_mesh.submesh_parent.topology_dm.getChart()
    coarse_subcell_to_point = get_entity_renumbering(
        coarse_mesh.topology_dm, coarse_mesh._cell_numbering, "cell",
    )[1]
    fine_subcell_to_point = get_entity_renumbering(
        fine_mesh.topology_dm, fine_mesh._cell_numbering, "cell",
    )[1]
    coarse_subpoint_is = coarse_mesh.topology_dm.getSubpointIS()
    fine_subpoint_is = fine_mesh.topology_dm.getSubpointIS()
    CHKERR(ISGetIndices(coarse_subpoint_is.iset, &coarse_subpoints))
    CHKERR(ISGetIndices(fine_subpoint_is.iset, &fine_subpoints))

    coarse_point_to_cell = np.full(
        coarse_parent_pEnd - coarse_parent_pStart, -1, dtype=IntType,
    )
    for c in range(ncoarse):
        coarse_point = coarse_subcell_to_point[c] + coarse_subStart
        parent = coarse_subpoints[coarse_point]
        if coarse_parent_pStart <= parent < coarse_parent_pEnd:
            coarse_point_to_cell[parent - coarse_parent_pStart] = c

    fine_parent_points = np.empty(nfine, dtype=IntType)
    for c in range(nfine):
        fine_point = fine_subcell_to_point[c] + fine_subStart
        fine_parent_points[c] = fine_subpoints[fine_point]

    fine_dm_points = np.asarray(fine_parent_points)
    if flgmaps[0] is not None:
        fno, fo = flgmaps
        fo.apply(fine_dm_points, result=fine_dm_points)
        fine_dm_points = fno.applyInverse(fine_dm_points,
                                          PETSc.LGMap.MapMode.MASK)

    CHKERR(DMPlexGetTransform(fine_dm.dm, &transform))
    if transform == NULL:
        raise RuntimeError("The fine DMPlex did not retain its refinement transform")
    coarse_dm_points = np.full(nfine, -1, dtype=IntType)
    for c in range(nfine):
        fine_point = fine_dm_points[c]
        if fine_point < 0:
            continue
        CHKERR(DMPlexTransformGetSourcePoint(
            transform, fine_point, NULL, NULL, &coarse_point, NULL,
        ))
        coarse_dm_points[c] = coarse_point

    if clgmaps[0] is not None:
        cno, co = clgmaps
        valid = np.asarray(coarse_dm_points) >= 0
        mapped_points = np.asarray(coarse_dm_points)[valid]
        cno.apply(mapped_points, result=mapped_points)
        np.asarray(coarse_dm_points)[valid] = co.applyInverse(
            mapped_points, PETSc.LGMap.MapMode.MASK,
        )

    fine_to_coarse = np.full((nfine, 1), -1, dtype=IntType)
    child_counts = np.zeros(ncoarse, dtype=IntType)
    for c in range(nfine):
        coarse_point = coarse_dm_points[c]
        if not (coarse_parent_pStart <= coarse_point < coarse_parent_pEnd):
            continue
        parent = coarse_point_to_cell[coarse_point - coarse_parent_pStart]
        if 0 <= parent < ncoarse:
            fine_to_coarse[c, 0] = parent
            child_counts[parent] += 1

    max_children = 0
    for c in range(ncoarse):
        if child_counts[c] > max_children:
            max_children = child_counts[c]
    max_children = fine_dm.comm.tompi4py().allreduce(max_children, op=MPI.MAX)
    coarse_to_fine = np.full((ncoarse, max_children), -1, dtype=IntType)
    child_counts[:] = 0
    for c in range(nfine):
        parent = fine_to_coarse[c, 0]
        if parent >= 0:
            coarse_to_fine[parent, child_counts[parent]] = c
            child_counts[parent] += 1

    CHKERR(ISRestoreIndices(coarse_subpoint_is.iset, &coarse_subpoints))
    CHKERR(ISRestoreIndices(fine_subpoint_is.iset, &fine_subpoints))
    return np.asarray(coarse_to_fine), np.asarray(fine_to_coarse)


@cython.boundscheck(False)
@cython.wraparound(False)
def preserved_points(PETSc.DM coarse_dm,
                     PETSc.Section coarse_cell_numbering,
                     PETSc.DM fine_dm,
                     PETSc.Section fine_cell_numbering):
    """Pair unrefined fine points with their coarse originals.

    Adaptive refinement copies an untouched coarse cell into the fine mesh
    without change. It therefore preserves the cone of every point in that
    cell, and so preserves the whole plex closure, point for point. PETSc's
    transform identifies the fine cells whose coarse cell was split, so its
    complement identifies the cells that are candidates for preservation.

    Parameters
    ----------
    coarse_dm : PETSc.DM
        The coarse mesh DMPlex.
    coarse_cell_numbering : PETSc.Section
        The cell numbering section of the coarse mesh.
    fine_dm : PETSc.DM
        The adaptively refined DMPlex.
    fine_cell_numbering : PETSc.Section
        The cell numbering section of the fine mesh.
    Returns
    -------
    numpy.ndarray
        An array over the chart of ``fine_dm``. For each fine point, it
        holds the coarse point it was copied from, or -1 if refinement
        changed that point.

    """
    cdef:
        PetscInt ncoarse, nfine, cStart, cEnd, pStart, pEnd
        PetscInt nunsplit, split_size, i, fine_cell, coarse_cell
        PetscInt fine_off, coarse_off
        PetscInt coarse_size, fine_size, c
        PetscInt *coarse_closure = NULL
        PetscInt *fine_closure = NULL
        const PetscInt *unsplit_cells = NULL
        PetscInt[::1] fine_to_coarse
        PETSc.PetscDMPlexTransform transform = NULL
        DMLabel split_label = NULL
        PETSc.PetscIS split_cell_is = NULL
        PETSc.PetscIS unsplit_cell_is = NULL

    ncoarse = num_owned_cells(coarse_dm)
    nfine = num_owned_cells(fine_dm)

    pStart, pEnd = fine_dm.getChart()
    fine_to_coarse = np.full(pEnd - pStart, -1, dtype=IntType)

    # The parent point relation is available when the coarse DM requested that
    # PETSc retain the transform that produced the fine DM.
    CHKERR(DMPlexGetTransform(fine_dm.dm, &transform))
    if transform == NULL:
        return np.asarray(fine_to_coarse)

    CHKERR(DMPlexTransformCreateSplitCellLabel(transform, fine_dm.dm, &split_label))
    cStart, cEnd = fine_dm.getHeightStratum(0)
    CHKERR(DMLabelGetStratumSize(split_label, 1, &split_size))
    if split_size:
        CHKERR(DMLabelGetStratumIS(split_label, 1, &split_cell_is))
        CHKERR(ISComplement(split_cell_is, cStart, cEnd, &unsplit_cell_is))
        CHKERR(ISGetSize(unsplit_cell_is, &nunsplit))
        CHKERR(ISGetIndices(unsplit_cell_is, &unsplit_cells))
    else:
        nunsplit = cEnd - cStart

    for i in range(nunsplit):
        fine_cell = unsplit_cells[i] if split_size else cStart + i
        CHKERR(PetscSectionGetOffset(fine_cell_numbering.sec, fine_cell, &fine_off))
        if not (0 <= fine_off < nfine):
            continue
        CHKERR(DMPlexTransformGetSourcePoint(
            transform, fine_cell, NULL, NULL, &coarse_cell, NULL,
        ))
        CHKERR(PetscSectionGetOffset(coarse_cell_numbering.sec, coarse_cell, &coarse_off))
        if not (0 <= coarse_off < ncoarse):
            continue
        CHKERR(DMPlexGetTransitiveClosure(coarse_dm.dm, coarse_cell, PETSC_TRUE,
                                          &coarse_size, &coarse_closure))
        CHKERR(DMPlexGetTransitiveClosure(fine_dm.dm, fine_cell, PETSC_TRUE,
                                          &fine_size, &fine_closure))
        if coarse_size == fine_size:
            for c in range(coarse_size):
                # Each closure interleaves a point with its orientation. Copy
                # a point only when its orientation matches in both meshes:
                # only then do the two cells order their nodes the same way.
                if coarse_closure[2*c + 1] == fine_closure[2*c + 1]:
                    fine_to_coarse[fine_closure[2*c] - pStart] = coarse_closure[2*c]
        CHKERR(DMPlexRestoreTransitiveClosure(coarse_dm.dm, coarse_cell, PETSC_TRUE,
                                              &coarse_size, &coarse_closure))
        CHKERR(DMPlexRestoreTransitiveClosure(fine_dm.dm, fine_cell, PETSC_TRUE,
                                              &fine_size, &fine_closure))

    if split_size:
        CHKERR(ISRestoreIndices(unsplit_cell_is, &unsplit_cells))
        CHKERR(ISDestroy(&unsplit_cell_is))
        CHKERR(ISDestroy(&split_cell_is))
    CHKERR(DMLabelDestroy(&split_label))
    return np.asarray(fine_to_coarse)


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
