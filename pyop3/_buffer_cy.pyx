"""Cython extensions for 'pyop3.buffer'.

This module should not be imported directly. Instead the functions defined here
should be exposed inside 'pyop3.buffer'.

"""
import numpy as np
from petsc4py import PETSc

from petsctools cimport cpetsc
from petsctools.cpetsc cimport CHKERR


def set_petsc_mat_diagonal(mat: cpetsc.Mat_py, value: cpetsc.PetscScalar) -> None:
    if mat.type == "nest":
        num_rows, num_columns = mat.getNestSize()
        for i in range(min(num_rows, num_columns)):
            submat = mat.getNestSubMatrix(i, i)
            set_petsc_mat_diagonal(submat, value)
    elif mat.type == "python":
        mat.getPythonContext().set_diagonal(value)
    else:
        _set_non_nested_petsc_mat_diagonal(mat, value)


def _set_non_nested_petsc_mat_diagonal(petscmat: cpetsc.Mat_py, value: cpetsc.PetscScalar) -> None:
    cdef:
        cpetsc.PetscInt    row_block_size_c, i_c, j_c
        cpetsc.PetscScalar *block_values_c = NULL

    row_block_size_c, _ = petscmat.block_sizes
    num_rows, _ = petscmat.local_size

    CHKERR(cpetsc.PetscCalloc1(row_block_size_c**2, &block_values_c))

    for i_c in range(row_block_size_c):
        for j_c in range(row_block_size_c):
            block_values_c[i_c*row_block_size_c+j_c] = value

    for i_c in range(num_rows // row_block_size_c):
        CHKERR(cpetsc.MatSetValuesBlockedLocal(petscmat.mat, 1, &i_c, 1, &i_c, block_values_c, cpetsc.INSERT_VALUES))

    CHKERR(cpetsc.PetscFree(block_values_c))


cdef extern from "petsc/private/matimpl.h":
    struct _p_Mat:
        void *data


ctypedef struct Mat_Preallocator:
    void *ht
    cpetsc.PetscInt *dnz
    cpetsc.PetscInt *onz


def get_preallocation(preallocator: cpetsc.Mat_py) -> tuple[PETSc.IntType, PETSc.IntType]:
    cdef:
        cpetsc.PetscInt nrow
        _p_Mat *A = <_p_Mat *>(preallocator.mat)
        Mat_Preallocator *p = <Mat_Preallocator *>(A.data)

    (nrow, _), _ = preallocator.sizes

    if p.dnz != NULL:
        dnz = <cpetsc.PetscInt[:nrow]>p.dnz
        dnz = np.asarray(dnz).copy()
    else:
        dnz = np.zeros(0, dtype=PETSc.IntType)
    if p.onz != NULL:
        onz = <cpetsc.PetscInt[:nrow]>p.onz
        onz = np.asarray(onz).copy()
    else:
        onz = np.zeros(0, dtype=PETSc.IntType)
    return dnz, onz
