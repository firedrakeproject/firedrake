import abc

from firedrake.preconditioners.base import PCBase
from firedrake.functionspace import FunctionSpace, MixedFunctionSpace
from firedrake.petsc import PETSc
from firedrake.ufl_expr import TestFunction, TrialFunction
import firedrake.dmhooks as dmhooks
from firedrake.dmhooks import get_function_space

import petsc4py.PETSc # in firedrake.petsc?

#outside: ksp.setOperators(A)

__all__ = ("OffloadPC")


class OffloadPC(PCBase): #still PETSc PC object?
    """Offload to GPU as PC to solve.

    Internally this makes a PETSc PC object that can be controlled by
    options using the extra options prefix ``offload_``.
    """

    _prefix = "offload_"

    def initialize(self, pc):
        A, P = pc.getOperators() #P preconditioner

        if pc.getType() != "assembled":
            raise ValueError("Expecting PC type assembled") #correct type?
        opc = pc #opc?
        appctx = self.get_appctx(pc) 
        fcp = appctx.get("form_compiler_parameters") 

        if P.type == "assembled": #not python value error - only assembled (preconditioner)
            context = P.getPythonContext()
            # It only makes sense to preconditioner/invert a diagonal
            # block in general.  That's all we're going to allow.
            if not context.on_diag: #still? diagonal block?
                raise ValueError("Only makes sense to invert diagonal block")
    
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix

        mat_type = PETSc.Options().getString(options_prefix + "mat_type", "aijcusparse") 

        (a, bcs) = self.form(pc, test, trial) 

        self.P = allocate_matrix(a, bcs=bcs, #eventually change allocate matrix
                                 form_compiler_parameters=fcp,
                                 mat_type=mat_type,
                                 options_prefix=options_prefix)

        # Transfer nullspace over
        Pmat = self.P.petscmat 
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)
        Pmat.setNearNullSpace(P.getNearNullSpace())

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -assembled_pc_type ksp.
        #?

#same - matrix here
        pc = PETSc.PC().create(comm=opc.comm) 
        pc.incrementTabLevel(1, parent=opc) #

        # We set a DM and an appropriate SNESContext on the constructed PC so one
        # can do e.g. multigrid or patch solves.
        dm = opc.getDM()
        self._ctx_ref = self.new_snes_ctx(opc, a, bcs, mat_type,
                                          fcp=fcp, options_prefix=options_prefix)

        pc.setDM(dm) 
        pc.setOptionsPrefix(options_prefix)

        #matrix to cuda
        A_cu = petsc4py.PETSc.Mat()
        A_cu.createDenseCUDA(A.petscmat.size)
        A.petscmat.copy(A_cu)
        A.petscmat = A_cu
        self._offload_A = A.petscmat #fishy

        P_cu = petsc4py.PETSc.Mat()
        P_cu.createDenseCUDA(A.petscmat.size)
        Pmat.petscmat.copy(P_cu)
        Pmat.petscmat = P_cu
        self._offload_A = Pmat.petscmat #fishy

        pc.setOperators(A_cu, P_cu)
        self.pc = pc
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False): 
            pc.setFromOptions()

    def update(self, pc):
        self._offload_A()

 #   def form(self, pc, test, trial):
 #       _, P = pc.getOperators()
 #       if P.getType() == "python":
 #           context = P.getPythonContext()
 #           return (context.a, context.row_bcs)
 #       else:
 #           context = dmhooks.get_appctx(pc.getDM())
 #           return (context.Jp or context.J, context._problem.bcs)

#vectors and solve
    def apply(self, pc, x, y): #y=b?
        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref), :
            b_cu = PETSc.Vec() 
            b_cu.createCUDAWithArrays(y)  
            u = PETSc.Vec()
            u.createCUDAWithArrays(x)
            self.pc.apply(x, y) #solve is here
            u.getArray() #give vector back

 #   def applyTranspose(self, pc, x, y): #same but other side
 #       dm = pc.getDM()
 #       with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
 #           b_cu = PETSc.Vec()
 #           b_cu.createCUDAWithArrays(y)  
 #           u = PETSc.Vec()
 #           u.createCUDAWithArrays(x)
 #           self.pc.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to solve on GPU\n")
            self.pc.view(viewer)