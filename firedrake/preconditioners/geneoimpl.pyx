IF WITH_GENEO:
    cimport petsc4py.PETSc as PETSc
    include "../dmplexinc.pxi"
    cdef extern from "geneo_c.h":
        int PCGenEOSetup(PETSc.PetscPC, PETSc.PetscMat, PETSc.PetscIS, PETSc.PetscIS*)
        int createGenEOPC(PETSc.PetscPC)

    def setup(PETSc.PC pc not None, PETSc.Mat localDirichlet, PETSc.IS dofmult not None, intersections not None):
        cdef PETSc.PetscIS *ises = NULL
        cdef PetscInt i, n
        cdef PETSc.PetscMat mat = NULL
        if localDirichlet is not None:
            mat = localDirichlet.mat

        n = len(intersections)
        CHKERR(PetscMalloc1(n, &ises))
        for i in range(n):
            ises[i] = (<PETSc.IS?>intersections[i]).iset
        CHKERR(PCGenEOSetup(pc.pc, mat, dofmult.iset, ises))
        CHKERR(PetscFree(ises))

    def register():
        CHKERR(PCRegister("geneo", &createGenEOPC))

    register()
ELSE:
    def setup(*args):
        raise NotImplementedError("firedrake must be installed with geneo4PETSc (--with-geneo)")

    def register():
        raise NotImplementedError("firedrake must be installed with geneo4PETSc (--with-geneo)") 
