from libc.stdint cimport uintptr_t

include "petschdr.pxi"

def set_patch_residual(patch, function, ctx, is_snes=False, interior_facets=False, exterior_facets=False):
    if is_snes:
        if interior_facets or exterior_facets:
            raise NotImplementedError
        CHKERR(SNESPatchSetComputeFunction((<PETSc.SNES?>patch).snes,
                                           <PetscPCPatchComputeFunction><uintptr_t>function,
                                           <void *><uintptr_t>ctx))
    else:
        if interior_facets:
            CHKERR(PCPatchSetComputeFunctionInteriorFacets((<PETSc.PC?>patch).pc,
                                                           <PetscPCPatchComputeFunction><uintptr_t>function,
                                                           <void *><uintptr_t>ctx))
        elif exterior_facets:
            CHKERR(PCPatchSetComputeFunctionExteriorFacets((<PETSc.PC?>patch).pc,
                                                           <PetscPCPatchComputeFunction><uintptr_t>function,
                                                           <void *><uintptr_t>ctx))
        else:
            CHKERR(PCPatchSetComputeFunction((<PETSc.PC?>patch).pc,
                                             <PetscPCPatchComputeFunction><uintptr_t>function,
                                             <void *><uintptr_t>ctx))


def set_patch_jacobian(patch, function, ctx, is_snes=False, interior_facets=False, exterior_facets=False):
    if is_snes:
        if interior_facets or exterior_facets:
            raise NotImplementedError
        CHKERR(SNESPatchSetComputeOperator((<PETSc.SNES?>patch).snes,
                                               <PetscPCPatchComputeOperator><uintptr_t>function,
                                               <void *><uintptr_t>ctx))
    else:
        if interior_facets:
            CHKERR(PCPatchSetComputeOperatorInteriorFacets((<PETSc.PC?>patch).pc,
                                                           <PetscPCPatchComputeOperator><uintptr_t>function,
                                                           <void *><uintptr_t>ctx))
        elif exterior_facets:
            CHKERR(PCPatchSetComputeOperatorExteriorFacets((<PETSc.PC?>patch).pc,
                                                           <PetscPCPatchComputeOperator><uintptr_t>function,
                                                           <void *><uintptr_t>ctx))
        else:
            CHKERR(PCPatchSetComputeOperator((<PETSc.PC?>patch).pc,
                                             <PetscPCPatchComputeOperator><uintptr_t>function,
                                             <void *><uintptr_t>ctx))
