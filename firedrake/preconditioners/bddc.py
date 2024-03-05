from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_function_space, get_appctx
from firedrake.ufl_expr import TestFunction, TrialFunction
from ufl import inner, div, dx, HCurl, HDiv
from pyop2.utils import as_tuple
import numpy

__all__ = ("BDDCPC",)


class BDDCPC(PCBase):
    """PC for PETSc PCBDDC"""

    _prefix = "bddc_"

    def initialize(self, pc):
        # Get context from pc
        _, P = pc.getOperators()
        dm = pc.getDM()
        self.prefix = pc.getOptionsPrefix() + self._prefix

        V = get_function_space(dm)

        # Create new PC object as BDDC type
        bddcpc = PETSc.PC().create(comm=pc.comm)
        bddcpc.incrementTabLevel(1, parent=pc)
        bddcpc.setOptionsPrefix(self.prefix)
        bddcpc.setOperators(*pc.getOperators())
        bddcpc.setType(PETSc.PC.Type.BDDC)

        ctx = get_appctx(dm)
        bcs = tuple(ctx._problem.bcs)
        if len(bcs) > 0:
            bc_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=False) for bc in bcs]))
            V.dof_dset.lgmap.apply(bc_nodes, result=bc_nodes)
            bndr = PETSc.IS().createGeneral(bc_nodes, comm=pc.comm)
            bddcpc.setBDDCDirichletBoundaries(bndr)

        appctx = self.get_appctx(pc)
        sobolev_space = V.ufl_element().sobolev_space
        if sobolev_space == HCurl:
            gradient = appctx.get("discrete_gradient", None)
            if gradient is None:
                from firedrake.preconditioners.fdm import tabulate_exterior_derivative
                Q = V.reconstruct(family="Lagrange")
                gradient = tabulate_exterior_derivative(Q, V)
            bddcpc.setBDDCDiscreteGradient(gradient)

        elif sobolev_space == HDiv:
            B = appctx.get("divergence_mat", None)
            if B is None:
                from firedrake.assemble import assemble
                degree = max(as_tuple(V.ufl_element().degree()))
                Q = V.reconstruct(family="DG", degree=degree-1, variant=None)
                b = inner(div(TrialFunction(V)), TestFunction(Q)) * dx
                B = assemble(b, mat_type="matfree")
            bddcpc.setBDDCDivergenceMat(B.petscmat)

        bddcpc.setFromOptions()
        self.pc = bddcpc

    def view(self, pc, viewer=None):
        self.pc.view(viewer=viewer)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)
