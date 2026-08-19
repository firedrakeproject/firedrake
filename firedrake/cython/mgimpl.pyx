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
def set_adaptive_parent_label(PETSc.DM coarse_dm,
                              PETSc.Section coarse_cell_numbering,
                              label_name):
    """Seed each coarse cell's own Firedrake cell number onto a DMPlex label.

    Must be called *before* refining ``coarse_dm``. Since the refinement
    transform propagates labels from a cell to its children, every cell of
    every subsequent refinement then carries the number of the coarse cell it
    descends from, which `adaptive_parent_child_cell_maps` reads back.

    Parameters
    ----------
    coarse_dm : PETSc.DM
        The coarse, pre-refinement, mesh DMPlex.
    coarse_cell_numbering : PETSc.Section
        The coarse mesh's cell numbering section.
    label_name : str
        Name of the label to create on ``coarse_dm`` and populate with each
        owned cell's Firedrake cell number. Any existing label of that name
        is discarded.

    """
    cdef:
        PetscInt ncoarse = num_owned_cells(coarse_dm)
        PetscInt cStart, cEnd, c, off
        DMLabel parent_label = NULL

    if coarse_dm.hasLabel(label_name):
        coarse_dm.removeLabel(label_name)
    coarse_dm.createLabel(label_name)
    label_name = label_name.encode()
    CHKERR(DMGetLabel(coarse_dm.dm, <const char*>label_name, &parent_label))
    cStart, cEnd = coarse_dm.getHeightStratum(0)
    for c in range(cStart, cEnd):
        CHKERR(PetscSectionGetOffset(coarse_cell_numbering.sec, c, &off))
        if 0 <= off < ncoarse:
            CHKERR(DMLabelSetValue(parent_label, c, off))


@cython.boundscheck(False)
@cython.wraparound(False)
def adaptive_parent_child_cell_maps(PETSc.DM coarse_dm,
                                    PETSc.DM fine_dm,
                                    PETSc.Section fine_cell_numbering,
                                    label_name):
    """Build Firedrake-numbered parent/child cell maps from a DMPlex label.

    ``fine_dm`` must be a DMPlex obtained by refining, however many times, the
    ``coarse_dm`` that `set_adaptive_parent_label` was seeded on.

    Parameters
    ----------
    coarse_dm : PETSc.DM
        The coarse, parent, mesh DMPlex.
    fine_dm : PETSc.DM
        The refined, child, mesh DMPlex.
    fine_cell_numbering : PETSc.Section
        The fine mesh's cell numbering section.
    label_name : str
        Name of the label on ``fine_dm``, propagated from
        `set_adaptive_parent_label`, mapping each fine cell to its coarse
        parent's Firedrake cell number.

    Returns
    -------
    tuple of numpy.ndarray
        The ``(coarse_to_fine, fine_to_coarse)`` Firedrake-numbered cell maps,
        padded with -1 where a coarse cell has fewer children than the
        busiest one.

    """
    cdef:
        PetscInt ncoarse = num_owned_cells(coarse_dm)
        PetscInt nfine = num_owned_cells(fine_dm)
        PetscInt cStart, cEnd, c, off, parent, i, stratum_size, max_children
        DMLabel parent_label = NULL
        PETSc.PetscIS stratum_is = NULL
        const PetscInt *stratum_points = NULL
        PetscInt[::1] child_counts
        PetscInt[:, ::1] coarse_to_fine
        PetscInt[:, ::1] fine_to_coarse

    label_name = label_name.encode()
    CHKERR(DMGetLabel(fine_dm.dm, <const char*>label_name, &parent_label))
    fine_to_coarse = np.full((nfine, 1), -1, dtype=IntType)
    child_counts = np.zeros(ncoarse, dtype=IntType)
    cStart, cEnd = fine_dm.getHeightStratum(0)
    # Walking by stratum (coarse cell) resolves each one through PETSc's O(1)
    # value -> stratum hash map and touches every fine cell exactly once, for
    # O(nfine + ncoarse) overall.
    for parent in range(ncoarse):
        CHKERR(DMLabelGetStratumSize(parent_label, parent, &stratum_size))
        if stratum_size <= 0:
            continue
        CHKERR(DMLabelGetStratumIS(parent_label, parent, &stratum_is))
        CHKERR(ISGetIndices(stratum_is, &stratum_points))
        for i in range(stratum_size):
            c = stratum_points[i]
            if not (cStart <= c < cEnd):
                continue
            CHKERR(PetscSectionGetOffset(fine_cell_numbering.sec, c, &off))
            if not (0 <= off < nfine):
                continue
            fine_to_coarse[off, 0] = parent
            child_counts[parent] += 1
        CHKERR(ISRestoreIndices(stratum_is, &stratum_points))
        CHKERR(ISDestroy(&stratum_is))

    # coarse_to_fine is rectangular, so every coarse cell's row must be wide
    # enough for its most prolific sibling. Different coarse cells can be
    # refined a different number of times, so this varies by process; take
    # the max across all ranks so the array shape agrees everywhere.
    max_children = 0
    for c in range(ncoarse):
        if child_counts[c] > max_children:
            max_children = child_counts[c]
    max_children = fine_dm.comm.tompi4py().allreduce(max_children, op=MPI.MAX)
    coarse_to_fine = np.full((ncoarse, max_children), -1, dtype=IntType)
    # Re-walk the fine cells (in Firedrake order this time, via
    # fine_to_coarse) appending each one to its parent's row. child_counts is
    # reused as a per-parent write cursor, reset to zero first.
    child_counts[:] = 0
    for c in range(nfine):
        parent = fine_to_coarse[c, 0]
        if parent >= 0:
            coarse_to_fine[parent, child_counts[parent]] = c
            child_counts[parent] += 1

    return np.asarray(coarse_to_fine), np.asarray(fine_to_coarse)


# Exposition:
#
# These next functions compute maps from coarse mesh cells to fine
# mesh cells and provide a consistent vertex reordering of each fine
# cell inside each coarse cell.  In parallel, this is somewhat
# complicated because the DMs only provide information about
# relationships between non-overlapped meshes, and we only have
# overlapped meshes.  We there need to translate non-overlapped DM
# numbering into overlapped-DM numbering and vice versa, as well as
# translating between firedrake numbering and DM numbering.
#
# A picture is useful here to make things clearer.
#
# To translate between overlapped and non-overlapped DM points, we
# need to go via global numbers (which don't change)
#
#      DM_orig<--.    ,-<--DM_new
#         |      |    |      |
#     L2G v  G2L ^    v L2G  ^ G2L
#         |      |    |      |
#         '-->-->Global-->---'
#
# Mapping between Firedrake numbering and DM numbering is carried out
# by computing the section permutation `get_entity_renumbering` above.
#
#            .->-o2n->-.
#      DM_new          Firedrake
#            `-<-n2o-<-'
#
# Finally, coarse to fine maps are produced on the non-overlapped DM
# and subsequently composed with the appropriate sequence of maps to
# get to Firedrake numbering (and vice versa).
#
#     DM_orig_coarse
#           |
#           v coarse_to_fine_cells [coarse_cell = floor(fine_cell / 2**tdim)]
#           |
#      DM_orig_fine
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def coarse_to_fine_cells(mc, mf, clgmaps, flgmaps):
    """Map the cells of a coarse mesh to those of its uniform refinement.

    Parameters
    ----------
    mc : MeshGeometry
        The coarse mesh.
    mf : MeshGeometry
        The fine mesh, obtained by uniformly refining the non-overlapped
        plex of ``mc``.
    clgmaps : tuple
        The coarse ``(non-overlapped, overlapped)`` point local-to-global maps.
    flgmaps : tuple
        The fine ``(non-overlapped, overlapped)`` point local-to-global maps.

    Returns
    -------
    numpy.ndarray
        Map from each owned coarse cell to the fine cells it was split into.
    numpy.ndarray
        Map from each owned fine cell to the coarse cell it came from.

    Notes
    -----
    Three numberings of the same cells meet here:

    1. Firedrake numbering, which lists owned cells before halo cells. The
       returned maps are indexed by, and contain, these numbers.
    2. Overlapped plex numbering, that of ``mesh.topology_dm``.
       `get_entity_renumbering` translates between 1 and 2.
    3. Non-overlapped plex numbering, that of the halo-free plex that was
       refined. Only here does the parent relation hold: uniform refinement
       splits cell ``p`` into cells ``p*nref`` to ``p*nref + nref - 1``.

    Applying an overlapped local-to-global map and then a non-overlapped
    global-to-local one translates between 2 and 3. Those two numberings are
    genuinely different orders, not a common prefix plus a halo: a plex built
    by ``DMPlexTransform`` (an adaptively refined one) numbers its cells by
    refinement case, so its owned cells are interleaved with its halo cells.
    """
    cdef:
        PETSc.DM cdm, fdm
        PetscInt cStart, cEnd, c, dim, nref, ncoarse
        PetscInt i, ccell, fcell, nfine
        np.ndarray coarse_to_fine
        np.ndarray fine_to_coarse
        np.ndarray co2n, fn2o, idx, found, permuted

    cdm = mc.topology_dm
    fdm = mf.topology_dm
    dim = cdm.getDimension()
    nref = <PetscInt> 2 ** dim
    ncoarse = mc.cell_set.size
    nfine = mf.cell_set.size
    # co2n: coarse overlapped plex cell -> coarse Firedrake cell
    # fn2o: fine Firedrake cell -> fine overlapped plex cell
    co2n, _ = get_entity_renumbering(cdm, mc._cell_numbering, "cell")
    _, fn2o = get_entity_renumbering(fdm, mf._cell_numbering, "cell")
    coarse_to_fine = np.full((ncoarse, nref), -1, dtype=PETSc.IntType)
    fine_to_coarse = np.full((nfine, 1), -1, dtype=PETSc.IntType)
    # Walk owned fine cells:
    cStart, cEnd = 0, nfine

    # In serial the overlapped and non-overlapped plexes are the same plex,
    # so both maps already speak the numbering the parent relation holds in.
    if mc.comm.size > 1:
        # Cells are the leading points of a plex chart, so these point maps
        # can be applied to cell numbers directly.
        cno, co = clgmaps
        fno, fo = flgmaps
        # Rebase fn2o onto the fine non-overlapped plex, one map per arrow:
        # fine Firedrake cell -> overlapped -> global -> non-overlapped.
        fo.apply(fn2o, result=fn2o)
        fn2o = fno.applyInverse(fn2o, PETSc.LGMap.MapMode.MASK)
        # Rebase co2n the same way, but here it is the *index* that changes
        # numbering, not the value, so send every local coarse cell through
        # the translation. MASK gives -1 for cells the non-overlapped plex
        # does not have.
        idx = np.arange(mc.cell_set.total_size, dtype=PETSc.IntType)
        co.apply(idx, result=idx)
        idx = cno.applyInverse(idx, PETSc.LGMap.MapMode.MASK)
        # idx[i] is where overlapped cell i lands, so scatter rather than
        # slice: the surviving cells need not be the leading ones.
        found = idx >= 0
        permuted = np.empty(found.sum(), dtype=PETSc.IntType)
        permuted[idx[found]] = co2n[found]
        co2n = permuted

    for c in range(cStart, cEnd):
        # Every owned fine cell exists on the non-overlapped plex.
        fcell = fn2o[c]
        assert 0 <= fcell < cEnd

        # Uniform refinement numbers the nref children of a cell
        # consecutively, so integer division recovers the parent.
        ccell = co2n[fcell // nref]
        assert 0 <= ccell < ncoarse
        fine_to_coarse[c, 0] = ccell
        for i in range(nref):
            if coarse_to_fine[ccell, i] == -1:
                coarse_to_fine[ccell, i] = c
                break
    return coarse_to_fine, fine_to_coarse


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
