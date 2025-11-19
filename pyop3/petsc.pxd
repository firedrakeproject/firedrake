# NOTE: This could realistically go into petsctools
"""This file basically exposes the PETSc API as a module for use in Cython."""
from petsc4py cimport PETSc as _PETSc

# clearer aliases from petsc4py, so the names here match the C API
ctypedef _PETSc.PetscMat Mat
ctypedef _PETSc.Mat Mat_py
ctypedef _PETSc.PetscSF PetscSF
ctypedef _PETSc.SF PetscSF_py
ctypedef _PETSc.PetscSection PetscSection
ctypedef _PETSc.Section PetscSection_py
ctypedef _PETSc.PetscIS IS
ctypedef _PETSc.IS IS_py

# other PETSc imports
from petsc4py.PETSc cimport (
    CHKERR,
    PetscErrorCode,
)


cdef extern from "petsc.h":
    # fundamental types
    ctypedef long PetscInt
    ctypedef double PetscReal
    ctypedef double PetscScalar
    ctypedef enum PetscBool:
        PETSC_TRUE
        PETSC_FALSE
    ctypedef enum InsertMode:
        INSERT_VALUES
        ADD_VALUES
    ctypedef enum PetscCopyMode:
        PETSC_COPY_VALUES
        PETSC_OWN_POINTER
        PETSC_USE_POINTER

    # memory management
    PetscErrorCode PetscCalloc1(size_t,void*)
    PetscErrorCode PetscMalloc1(size_t,void*)
    PetscErrorCode PetscFree(void*)

    # Mat
    PetscErrorCode MatSetValuesBlockedLocal(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode)

    # PetscSF
    ctypedef struct PetscSFNode:
        pass

    PetscErrorCode PetscSFGetGraph(PetscSF,PetscInt*,PetscInt*,PetscInt**,PetscSFNode**)
    PetscErrorCode PetscSFSetGraph(PetscSF,PetscInt,PetscInt,PetscInt*,PetscCopyMode,PetscSFNode*,PetscCopyMode)

    # PetscSection
    PetscErrorCode PetscSectionGetDof(PetscSection,PetscInt,PetscInt*)
    PetscErrorCode PetscSectionSetDof(PetscSection,PetscInt,PetscInt)
    PetscErrorCode PetscSectionGetOffset(PetscSection,PetscInt,PetscInt*)
