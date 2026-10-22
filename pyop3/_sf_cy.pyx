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
    sf: cpetsc.PetscSF_py,
    const cpetsc.PetscInt[::1] selected_points,
    p_start: cpetsc.PetscInt,
    p_end: cpetsc.PetscInt,
) -> cpetsc.PetscSF_py:
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

        cpetsc.PetscInt nroots, nleaves
        const cpetsc.PetscInt *ilocal = NULL
        const cpetsc.PetscSFNode *iremote = NULL
        cpetsc.PetscInt pStart, pEnd, p

    point_sf = sf
    local_sec = section
    CHKERR_c(cpetsc.PetscSFGetGraph(point_sf.sf, &nroots, &nleaves, &ilocal, &iremote))
    pStart, pEnd = local_sec.getChart()
    assert pEnd - pStart == nroots, f"pEnd - pStart ({pEnd - pStart}) != nroots ({nroots})"
    assert pStart == 0
    CHKERR_c(cpetsc.PetscMalloc1(pEnd-pStart, &local_offsets))
    CHKERR_c(cpetsc.PetscMalloc1(pEnd-pStart, &remote_offsets))
    for p in range(pStart, pEnd):
        CHKERR_c(cpetsc.PetscSectionGetOffset(local_sec.sec, p, &local_offsets[p]))
    cdef MPI.Datatype unit = MPI._typedict[np.dtype(IntType).char]
    CHKERR_c(cpetsc.PetscSFBcastBegin(point_sf.sf, <cmpi.MPI_Datatype>unit.ob_mpi, local_offsets, remote_offsets, cmpi.MPI_REPLACE))
    CHKERR_c(cpetsc.PetscSFBcastEnd(point_sf.sf, <cmpi.MPI_Datatype>unit.ob_mpi, local_offsets, remote_offsets, cmpi.MPI_REPLACE))

    # PetscSFCreateSectionSF() creates the SF, so the wrapper must be empty
    halo_exchange_sf = PETSc.SF()
    CHKERR_c(cpetsc.PetscSFCreateSectionSF(sf.sf, section.sec, remote_offsets, section.sec, &halo_exchange_sf.sf))
    CHKERR_c(cpetsc.PetscFree(local_offsets))
    CHKERR_c(cpetsc.PetscFree(remote_offsets))
    return halo_exchange_sf


def renumber_petsc_sf(sf: cpetsc.PetscSF_py, renumbering: cpetsc.PetscInt[::1]) -> cpetsc.PetscSF_py:
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
