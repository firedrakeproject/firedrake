"""Cython extensions for 'pyop3.axis_tree.tree'.

This module should not be imported directly. Instead the functions defined here
should be exposed inside 'pyop3.axis_tree.tree'.

"""
import numpy as np
import pyop3.dtypes

from mpi4py import MPI

from petsctools cimport cpetsc
from petsctools.cpetsc cimport CHKERR
from petsc4py cimport PETSc


def create_section_with_arrays(
    comm: MPI.Comm,
    const cpetsc.PetscInt[::1] sizes,
    const cpetsc.PetscInt[::1] offsets=None,
) -> cpetsc.PetscSection_py:
    cdef cpetsc.PetscSection_py sec

    sec = PETSc.Section().create(comm=comm)
    npoints = len(sizes)
    sec.setChart(0, npoints)

    if offsets is None:
        for pt in range(npoints):
            CHKERR(cpetsc.PetscSectionSetDof(sec.sec, pt, sizes[pt]))
        sec.setUp()
    else:
        assert len(sizes) == len(offsets)
        for pt in range(npoints):
            CHKERR(cpetsc.PetscSectionSetDof(sec.sec, pt, sizes[pt]))
            CHKERR(cpetsc.PetscSectionSetOffset(sec.sec, pt, offsets[pt]))
    return sec
