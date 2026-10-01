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

from mpi4py cimport libmpi as cmpi
from mpi4py cimport MPI
from petsctools cimport cpetsc
from petsctools.cpetsc cimport CHKERR as CHKERR_c



def filter_petsc_sf(
    sf: cpetsc.PetscSF_py,
    selected_points: np_c.ndarray[IntType],  # TODO: IS?
    p_start: cpetsc.PetscInt,
    p_end: cpetsc.PetscInt,
) -> cpetsc.PetscSF_py:
    """
    neednt be ordered

    but must be unique

    """
    cdef:
        cpetsc.PetscSF_py     sf_filtered
        cpetsc.PetscSection_py section

        cpetsc.PetscInt      npoints_c, i_c, p_c
        cpetsc.PetscInt      *remoteOffsets_c = NULL

    npoints_c = len(selected_points)
    if npoints_c > 0:
        utils.debug_assert(lambda: p_start <= min(selected_points))
        utils.debug_assert(lambda: p_end >= max(selected_points))
        utils.debug_assert(lambda: utils.has_unique_entries(selected_points))

    section = PETSc.Section().create(comm=sf.comm)
    section.setChart(p_start, p_end)
    for i_c in range(npoints_c):
        p_c = selected_points[i_c]
        CHKERR_c(cpetsc.PetscSectionSetDof(section.sec, p_c, 1))
    section.setUp()

    return create_petsc_section_sf(sf, section)


def create_petsc_section_sf(sf: cpetsc.PetscSF_py, section: cpetsc.PetscSection_py) -> PETSc.SF:
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
        cpetsc.PetscSF_py point_sf, halo_exchange_sf
        cpetsc.PetscSection_py local_sec
        cpetsc.PetscInt *local_offsets = NULL
        cpetsc.PetscInt *remote_offsets = NULL

        cpetsc.PetscInt dof_nroots, dof_nleaves
        cpetsc.PetscInt *dof_ilocal = NULL
        cpetsc.PetscSFNode *dof_iremote = NULL
        cpetsc.PetscInt nroots, nleaves
        const cpetsc.PetscInt *ilocal = NULL
        const cpetsc.PetscSFNode *iremote = NULL
        cpetsc.PetscInt pStart, pEnd, p, dof, off, m, n, i, j

    point_sf = sf
    local_sec = section
    CHKERR_c(cpetsc.PetscSFGetGraph(point_sf.sf, &nroots, &nleaves, &ilocal, &iremote))
    pStart, pEnd = local_sec.getChart()
    assert pEnd - pStart == nroots, f"pEnd - pStart ({pEnd - pStart}) != nroots ({nroots})"
    assert pStart == 0
    m = 0
    CHKERR_c(cpetsc.PetscMalloc1(pEnd-pStart, &local_offsets))
    CHKERR_c(cpetsc.PetscMalloc1(pEnd-pStart, &remote_offsets))  # fill with -1s
    for p in range(pStart, pEnd):
        remote_offsets[p] = -1
    # local_offsets = np.empty(pEnd - pStart, dtype=IntType)
    # remote_offsets = np.full(pEnd - pStart, -1, dtype=IntType)
    for p in range(pStart, pEnd):
        CHKERR_c(cpetsc.PetscSectionGetDof(local_sec.sec, p, &dof))
        CHKERR_c(cpetsc.PetscSectionGetOffset(local_sec.sec, p, &local_offsets[p]))
        m += dof
    cdef MPI.Datatype unit = MPI._typedict[np.dtype(IntType).char]
    CHKERR_c(cpetsc.PetscSFBcastBegin(point_sf.sf, <cmpi.MPI_Datatype>unit.ob_mpi, local_offsets, remote_offsets, cmpi.MPI_REPLACE))
    CHKERR_c(cpetsc.PetscSFBcastEnd(point_sf.sf, <cmpi.MPI_Datatype>unit.ob_mpi, local_offsets, remote_offsets, cmpi.MPI_REPLACE))

    halo_exchange_sf = PETSc.SF().create(comm=point_sf.comm)
    CHKERR_c(cpetsc.PetscSFCreateSectionSF(sf.sf, section.sec, remote_offsets, section.sec, &halo_exchange_sf.sf))
    return halo_exchange_sf


def renumber_petsc_sf(sf: cpetsc.PetscSF_py, renumbering: cpetsc.IS_py) -> cpetsc.PetscSF_py:
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
        cpetsc.PetscSF_py      sf_renum
        cpetsc.PetscSection_py section

        cpetsc.PetscInt      npoints_c, p_c
        cpetsc.PetscInt      *remoteOffsets_c = NULL

    npoints_c = renumbering.getLocalSize()

    # section = PETSc.Section().create(sf.comm)
    section = PETSc.Section().create(MPI.COMM_SELF)
    section.setChart(0, npoints_c)
    for p_c in range(npoints_c):
        CHKERR_c(cpetsc.PetscSectionSetDof(section.sec, p_c, 1))
    section.setPermutation(renumbering)
    section.setUp()

    return create_petsc_section_sf(sf, section)
