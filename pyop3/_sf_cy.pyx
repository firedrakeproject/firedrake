"""Cython extensions for 'pyop3.sf'.

This module should not be imported directly. Instead the functions defined here
should be exposed inside 'pyop3.sf'.

"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from pyop3 import utils
from pyop3.dtypes import IntType
# ---
cimport numpy as np_c

from pyop3 cimport petsc as petsc_c
from pyop3.petsc cimport CHKERR as CHKERR_c


def filter_petsc_sf(
    sf: petsc_c.PetscSF_py,
    selected_points: np_c.ndarray[IntType],  # TODO: IS?
    p_start: petsc_c.PetscInt,
    p_end: petsc_c.PetscInt,
) -> petsc_c.PetscSF_py:
    """
    neednt be ordered

    but must be unique

    """
    cdef:
        petsc_c.PetscSF_py     sf_filtered
        petsc_c.PetscSection_py section

        petsc_c.PetscInt      npoints_c, i_c, p_c
        petsc_c.PetscInt      *remoteOffsets_c = NULL

    npoints_c = len(selected_points)
    if npoints_c > 0:
        utils.debug_assert(lambda: p_start <= min(selected_points))
        utils.debug_assert(lambda: p_end >= max(selected_points))
        utils.debug_assert(lambda: utils.has_unique_entries(selected_points))

    section = PETSc.Section().create(comm=sf.comm)
    section.setChart(p_start, p_end)
    for i_c in range(npoints_c):
        p_c = selected_points[i_c]
        CHKERR_c(petsc_c.PetscSectionSetDof(section.sec, p_c, 1))
    section.setUp()

    return create_petsc_section_sf(sf, section)


def create_petsc_section_sf(sf: petsc_c.PetscSF_py, section: petsc_c.PetscSection_py) -> PETSc.SF:
    """Create the halo exchange sf.

    Parameters
    ----------
    dm : PETSc.DM
        The section dm.

    Returns
    -------
    PETSc.SF
        The halo exchange sf.

    Notes
    -----
    The output sf is to update all ghost DoFs including constrained ones if any.

    """
    cdef:
        petsc_c.PetscSF_py point_sf, halo_exchange_sf
        petsc_c.PetscSection_py local_sec
        np_c.ndarray local_offsets
        np_c.ndarray remote_offsets

        petsc_c.PetscInt dof_nroots, dof_nleaves
        petsc_c.PetscInt *dof_ilocal = NULL
        petsc_c.PetscSFNode *dof_iremote = NULL
        petsc_c.PetscInt nroots, nleaves
        const petsc_c.PetscInt *ilocal = NULL
        const petsc_c.PetscSFNode *iremote = NULL
        petsc_c.PetscInt pStart, pEnd, p, dof, off, m, n, i, j

    point_sf = sf
    local_sec = section
    CHKERR_c(petsc_c.PetscSFGetGraph(point_sf.sf, &nroots, &nleaves, &ilocal, &iremote))
    pStart, pEnd = local_sec.getChart()
    assert pEnd - pStart == nroots, f"pEnd - pStart ({pEnd - pStart}) != nroots ({nroots})"
    assert pStart == 0
    m = 0
    local_offsets = np.empty(pEnd - pStart, dtype=IntType)
    remote_offsets = np.full(pEnd - pStart, -1, dtype=IntType)
    for p in range(pStart, pEnd):
        CHKERR_c(petsc_c.PetscSectionGetDof(local_sec.sec, p, &dof))
        CHKERR_c(petsc_c.PetscSectionGetOffset(local_sec.sec, p, &off))
        local_offsets[p] = off
        m += dof
    unit = MPI._typedict[np.dtype(IntType).char]
    point_sf.bcastBegin(unit, local_offsets, remote_offsets, MPI.REPLACE)
    point_sf.bcastEnd(unit, local_offsets, remote_offsets, MPI.REPLACE)
    n = 0
    # ilocal == NULL if local leaf points are [0, 1, 2, ...).
    for i in range(nleaves):
        p = ilocal[i] if ilocal else i
        CHKERR_c(petsc_c.PetscSectionGetDof(local_sec.sec, p, &dof))
        n += dof
    CHKERR_c(petsc_c.PetscMalloc1(n, &dof_ilocal))
    CHKERR_c(petsc_c.PetscMalloc1(n, &dof_iremote))
    n = 0
    for i in range(nleaves):
        # ilocal == NULL if local leaf points are [0, 1, 2, ...).
        p = ilocal[i] if ilocal else i
        assert remote_offsets[p] >= 0
        CHKERR_c(petsc_c.PetscSectionGetDof(local_sec.sec, p, &dof))
        CHKERR_c(petsc_c.PetscSectionGetOffset(local_sec.sec, p, &off))
        for j in range(dof):
            dof_ilocal[n] = off + j
            dof_iremote[n].rank = iremote[i].rank
            dof_iremote[n].index = remote_offsets[p] + j
            n += 1
    halo_exchange_sf = PETSc.SF().create(comm=point_sf.comm)
    CHKERR_c(petsc_c.PetscSFSetGraph(halo_exchange_sf.sf, m, n, dof_ilocal, petsc_c.PETSC_OWN_POINTER, dof_iremote, petsc_c.PETSC_OWN_POINTER))
    return halo_exchange_sf


def renumber_petsc_sf(sf: petsc_c.PetscSF_py, renumbering: petsc_c.IS_py) -> petsc_.PetscSF_py:
    """Renumber an SF.

    Parameters
    ----------
    sf :
        The input SF.
    renumbering :
        The renumbering to apply.

    Returns
    -------
    PETSc.SF :
        The renumbered SF.

    Notes
    -----
    To renumber the SF we create a Section containing 1 DoF per point, set
    its permutation, and then call ``PetscSFCreateSectionSF()``.

    """
    cdef:
        petsc_c.PetscSF_py      sf_renum
        petsc_c.PetscSection_py section

        petsc_c.PetscInt      npoints_c, p_c
        petsc_c.PetscInt      *remoteOffsets_c = NULL

    npoints_c = renumbering.getLocalSize()

    # section = PETSc.Section().create(sf.comm)
    section = PETSc.Section().create(MPI.COMM_SELF)
    section.setChart(0, npoints_c)
    for p_c in range(npoints_c):
        CHKERR_c(petsc_c.PetscSectionSetDof(section.sec, p_c, 1))
    section.setPermutation(renumbering)
    section.setUp()

    return create_petsc_section_sf(sf, section)


