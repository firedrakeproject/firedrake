"""Cython extensions for 'pyop3.buffer'.

This module should not be imported directly. Instead the functions defined here
should be exposed inside 'pyop3.buffer'.

"""
from petsc4py import PETSc

from pyop3 cimport petsc as petsc_c
from pyop3.petsc cimport CHKERR as CHKERR_c


def set_petsc_mat_diagonal(mat: petsc_c.Mat_py, value: petsc_c.PetscScalar) -> None:
    if mat.type == "nest":
        num_rows, num_columns = mat.getNestSize()
        for i in range(min(num_rows, num_columns)):
            submat = mat.getNestSubMatrix(i, i)
            set_petsc_mat_diagonal(submat, value)
    else:
        _set_non_nested_petsc_mat_diagonal(mat, value)


def _set_non_nested_petsc_mat_diagonal(petscmat: petsc_c.Mat_py, value: petsc_c.PetscScalar) -> None:
    cdef:
        petsc_c.PetscInt    row_block_size_c, i_c, j_c
        petsc_c.PetscScalar *block_values_c = NULL

    row_block_size_c, _ = petscmat.block_sizes
    num_rows, _ = petscmat.local_size

    CHKERR_c(petsc_c.PetscCalloc1(row_block_size_c**2, &block_values_c))

    for i_c in range(row_block_size_c):
        for j_c in range(row_block_size_c):
            block_values_c[i_c*row_block_size_c+j_c] = value

    for i_c in range(num_rows // row_block_size_c):
        CHKERR_c(petsc_c.MatSetValuesBlockedLocal(petscmat.mat, 1, &i_c, 1, &i_c, block_values_c, petsc_c.INSERT_VALUES))

    CHKERR_c(petsc_c.PetscFree(block_values_c))
