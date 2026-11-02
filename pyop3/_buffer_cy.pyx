"""Cython extensions for 'pyop3.buffer'.

This module should not be imported directly. Instead the functions defined here
should be exposed inside 'pyop3.buffer'.

"""
import numbers

from petsctools cimport cpetsc
from petsctools.cpetsc cimport CHKERR


cdef extern from "<petsc4py/pyscalar.h>":
    object      PyPetscScalar_FromPetscScalar(cpetsc.PetscScalar)
    cpetsc.PetscScalar PyPetscScalar_AsPetscScalar(object) except? <cpetsc.PetscScalar>-1.0


def set_petsc_mat_diagonal(mat: cpetsc.Mat_py, value: cpetsc.PetscScalar[::1] | cpetsc.PetscScalar) -> None:
    if mat.type == "nest":
        assert isinstance(value, numbers.Number)
        num_rows, num_columns = mat.getNestSize()
        for i in range(min(num_rows, num_columns)):
            submat = mat.getNestSubMatrix(i, i)
            set_petsc_mat_diagonal(submat, value)
    elif mat.type == "python":
        mat.getPythonContext().set_diagonal(value)
    else:
        _set_non_nested_petsc_mat_diagonal(mat, value)


def _set_non_nested_petsc_mat_diagonal(petscmat: cpetsc.Mat_py, value: cpetsc.PetscScalar[::1] | cpetsc.PetscScalar) -> None:
    if isinstance(value, numbers.Number):
        _set_non_nested_petsc_mat_diagonal_scalar(petscmat, value)
    else:
        _set_non_nested_petsc_mat_diagonal_vector(petscmat, value)


def _set_non_nested_petsc_mat_diagonal_scalar(petscmat: cpetsc.Mat_py, value) -> None:
    cdef:
        cpetsc.PetscInt    row_block_size_c, i_c
        cpetsc.PetscScalar *block_values_c = NULL

    row_block_size_c, _ = petscmat.block_sizes
    num_rows, _ = petscmat.local_size

    CHKERR(cpetsc.PetscCalloc1(row_block_size_c**2, &block_values_c))

    for i_c in range(row_block_size_c):
        block_values_c[i_c*row_block_size_c+i_c] = PyPetscScalar_AsPetscScalar(value)
    for i_c in range(num_rows // row_block_size_c):
        CHKERR(cpetsc.MatSetValuesBlockedLocal(petscmat.mat, 1, &i_c, 1, &i_c, block_values_c, cpetsc.INSERT_VALUES))

    CHKERR(cpetsc.PetscFree(block_values_c))


def _set_non_nested_petsc_mat_diagonal_vector(petscmat: cpetsc.Mat_py, value) -> None:
    cdef:
        cpetsc.PetscInt row_block_size_c, i_c, j_c
        cpetsc.PetscScalar *block_values_c = NULL

    row_block_size_c, _ = petscmat.block_sizes
    num_rows, _ = petscmat.local_size

    CHKERR(cpetsc.PetscCalloc1(row_block_size_c**2, &block_values_c))

    for i_c in range(num_rows // row_block_size_c):
        for j_c in range(row_block_size_c):
            block_values_c[j_c*row_block_size_c+j_c] = PyPetscScalar_AsPetscScalar(value[i_c])

        CHKERR(cpetsc.MatSetValuesBlockedLocal(petscmat.mat, 1, &i_c, 1, &i_c, block_values_c, cpetsc.INSERT_VALUES))

    CHKERR(cpetsc.PetscFree(block_values_c))
