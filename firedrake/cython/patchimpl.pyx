from libc.stdint cimport uintptr_t

include "petschdr.pxi"


def pcpatch_set_compute_operator(patch, function, ctx):
    CHKERR(PCPatchSetComputeOperator((<PETSc.PC?>patch).pc,
                                     <PetscPCPatchComputeOperator><uintptr_t>function,
                                     <void *><uintptr_t>ctx))


def pcpatch_set_compute_operator_interior_facets(patch, function, ctx):
    CHKERR(PCPatchSetComputeOperatorInteriorFacets((<PETSc.PC?>patch).pc,
                                                   <PetscPCPatchComputeOperator><uintptr_t>function,
                                                   <void *><uintptr_t>ctx))


def pcpatch_set_compute_operator_exterior_facets(patch, function, ctx):
    CHKERR(PCPatchSetComputeOperatorExteriorFacets((<PETSc.PC?>patch).pc,
                                                   <PetscPCPatchComputeOperator><uintptr_t>function,
                                                   <void *><uintptr_t>ctx))


def pcpatch_set_compute_function(patch, function, ctx):
    CHKERR(PCPatchSetComputeFunction((<PETSc.PC?>patch).pc,
                                     <PetscPCPatchComputeFunction><uintptr_t>function,
                                     <void *><uintptr_t>ctx))


def pcpatch_set_compute_function_interior_facets(patch, function, ctx):
    CHKERR(PCPatchSetComputeFunctionInteriorFacets((<PETSc.PC?>patch).pc,
                                                   <PetscPCPatchComputeFunction><uintptr_t>function,
                                                   <void *><uintptr_t>ctx))


def pcpatch_set_compute_function_exterior_facets(patch, function, ctx):
    CHKERR(PCPatchSetComputeFunctionExteriorFacets((<PETSc.PC?>patch).pc,
                                                   <PetscPCPatchComputeFunction><uintptr_t>function,
                                                   <void *><uintptr_t>ctx))


def snespatch_set_compute_operator(patch, function, ctx):
    CHKERR(SNESPatchSetComputeOperator((<PETSc.SNES?>patch).snes,
                                        <PetscPCPatchComputeOperator><uintptr_t>function,
                                        <void *><uintptr_t>ctx))


def snespatch_set_compute_function(patch, function, ctx):
    CHKERR(SNESPatchSetComputeFunction((<PETSc.SNES?>patch).snes,
                                       <PetscPCPatchComputeFunction><uintptr_t>function,
                                       <void *><uintptr_t>ctx))
