"""Cython extensions for 'pyop3.sf'.

This module should not be imported directly. Instead the functions defined here
should be exposed inside 'pyop3.sf'.

"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from pyop3.dtypes import IntType

cimport cython

from mpi4py cimport libmpi as cmpi
from mpi4py cimport MPI
from petsctools cimport cpetsc
from petsctools.cpetsc cimport CHKERR as CHKERR_c


@cython.boundscheck(False)
@cython.wraparound(False)
def filter_petsc_sf(
    cpetsc.PetscSF_py sf,
    const cpetsc.PetscInt[::1] selected_points,
    cpetsc.PetscInt p_start,
    cpetsc.PetscInt p_end,
):
    """
    neednt be ordered

    but must be unique

    """
    cdef:
        cpetsc.PetscSection_py section

        cpetsc.PetscInt npoints_c, i_c, p_c

    npoints_c = len(selected_points)

    section = PETSc.Section().create(comm=sf.comm)
    section.setChart(p_start, p_end)
    for i_c in range(npoints_c):
        p_c = selected_points[i_c]
        cpetsc.PetscSectionSetDof(section.sec, p_c, 1)
    section.setUp()

    return create_petsc_section_sf(sf, section)


def create_petsc_section_sf(cpetsc.PetscSF_py sf, cpetsc.PetscSection_py section) -> PETSc.SF:
    """Create the halo exchange sf.

    Parameters
    ----------
    sf : PETSc.SF
        The point sf.

    section : PETSc.Section
        The local section.

    Returns
    -------
    PETSc.SF
        The halo exchange sf. Its graph is set in this function but the actual SF is set up on its
        first communication call.

    Notes
    -----
    The output sf is to update all ghost DoFs including constrained ones if any.

    """
    cdef:
        cpetsc.PetscSF_py point_sf, halo_exchange_sf
        cpetsc.PetscSection_py local_sec
        cpetsc.PetscInt *local_offsets = NULL
        cpetsc.PetscInt *remote_offsets = NULL
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
    CHKERR_c(cpetsc.PetscMalloc1(pEnd-pStart, &local_offsets))
    CHKERR_c(cpetsc.PetscMalloc1(pEnd-pStart, &remote_offsets))
    m = 0
    for p in range(pStart, pEnd):
        CHKERR_c(cpetsc.PetscSectionGetDof(local_sec.sec, p, &dof))
        CHKERR_c(cpetsc.PetscSectionGetOffset(local_sec.sec, p, &off))
        local_offsets[p] = off
        remote_offsets[p] = -1
        m += dof
    cdef MPI.Datatype unit = MPI._typedict[np.dtype(IntType).char]
    CHKERR_c(cpetsc.PetscSFBcastBegin(point_sf.sf, <cmpi.MPI_Datatype>unit.ob_mpi, local_offsets, remote_offsets, cmpi.MPI_REPLACE))
    CHKERR_c(cpetsc.PetscSFBcastEnd(point_sf.sf, <cmpi.MPI_Datatype>unit.ob_mpi, local_offsets, remote_offsets, cmpi.MPI_REPLACE))

    n = 0
    # ilocal == NULL if local leaf points are [0, 1, 2, ...).
    for i in range(nleaves):
        p = ilocal[i] if ilocal else i
        CHKERR_c(cpetsc.PetscSectionGetDof(local_sec.sec, p, &dof))
        n += dof
    CHKERR_c(cpetsc.PetscMalloc1(n, &dof_ilocal))
    CHKERR_c(cpetsc.PetscMalloc1(n, &dof_iremote))
    n = 0
    for i in range(nleaves):
        # ilocal == NULL if local leaf points are [0, 1, 2, ...).
        p = ilocal[i] if ilocal else i
        assert remote_offsets[p] >= 0
        CHKERR_c(cpetsc.PetscSectionGetDof(local_sec.sec, p, &dof))
        CHKERR_c(cpetsc.PetscSectionGetOffset(local_sec.sec, p, &off))
        for j in range(dof):
            dof_ilocal[n] = off + j
            dof_iremote[n].rank = iremote[i].rank
            dof_iremote[n].index = remote_offsets[p] + j
            n += 1
    CHKERR_c(cpetsc.PetscFree(local_offsets))
    CHKERR_c(cpetsc.PetscFree(remote_offsets))

    halo_exchange_sf = PETSc.SF().create(comm=point_sf.comm)
    CHKERR_c(cpetsc.PetscSFSetGraph(halo_exchange_sf.sf, m, n, dof_ilocal, cpetsc.PETSC_OWN_POINTER, dof_iremote, cpetsc.PETSC_OWN_POINTER))
    return halo_exchange_sf


def renumber_petsc_sf(cpetsc.PetscSF_py sf, cpetsc.PetscInt[::1] renumbering):
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
        cpetsc.PetscSection_py section

        cpetsc.PetscInt npoints_c, p_c

    npoints_c = len(renumbering)

    # section = PETSc.Section().create(sf.comm)
    section = PETSc.Section().create(MPI.COMM_SELF)
    section.setChart(0, npoints_c)
    for p_c in range(npoints_c):
        CHKERR_c(cpetsc.PetscSectionSetDof(section.sec, p_c, 1))

    perm = PETSc.IS().createGeneral(renumbering, comm=MPI.COMM_SELF)
    section.setPermutation(perm)
    section.setUp()

    return create_petsc_section_sf(sf, section)
