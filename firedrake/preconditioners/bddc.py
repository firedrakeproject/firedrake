from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.facet_split import restrict, restricted_local_dofs
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_function_space, get_appctx
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace
from ufl import curl, div, HCurl, HDiv, inner, dx
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

        opts = PETSc.Options(bddcpc.getOptionsPrefix())
        if V.ufl_element().variant() == "fdm" and "pc_bddc_use_local_mat_graph" not in opts:
            # Disable computation of disconected components of subdomain interfaces
            opts["pc_bddc_use_local_mat_graph"] = False

        ctx = get_appctx(dm)
        bcs = tuple(ctx._problem.bcs)
        if V.extruded:
            boundary_nodes = numpy.unique(numpy.concatenate(list(map(V.boundary_nodes, ("on_boundary", "top", "bottom")))))
        else:
            boundary_nodes = V.boundary_nodes("on_boundary")
        if len(bcs) == 0:
            dir_nodes = numpy.empty(0, dtype=boundary_nodes.dtype)
        else:
            dir_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=False) for bc in bcs]))
        neu_nodes = numpy.setdiff1d(boundary_nodes, dir_nodes)

        V.dof_dset.lgmap.apply(dir_nodes, result=dir_nodes)
        dir_bndr = PETSc.IS().createGeneral(dir_nodes, comm=pc.comm)
        bddcpc.setBDDCDirichletBoundaries(dir_bndr)

        V.dof_dset.lgmap.apply(neu_nodes, result=neu_nodes)
        neu_bndr = PETSc.IS().createGeneral(neu_nodes, comm=pc.comm)
        bddcpc.setBDDCNeumannBoundaries(neu_bndr)

        appctx = self.get_appctx(pc)
        sobolev_space = V.ufl_element().sobolev_space

        tdim = V.mesh().topological_dimension()
        degree = max(as_tuple(V.ufl_element().degree()))
        if tdim >= 2 and V.finat_element.formdegree == tdim-1:
            B = appctx.get("divergence_mat", None)
            if B is None:
                from firedrake.assemble import assemble
                d = {HCurl: curl, HDiv: div}[sobolev_space]
                if V.shape == ():
                    make_function_space = FunctionSpace
                elif len(V.shape) == 1:
                    make_function_space = VectorFunctionSpace
                else:
                    make_function_space = TensorFunctionSpace
                Q = make_function_space(V.mesh(), "DG", degree-1)
                b = inner(d(TrialFunction(V)), TestFunction(Q)) * dx(degree=2*(degree-1))
                B = assemble(b, mat_type="matfree")
            bddcpc.setBDDCDivergenceMat(B.petscmat)
        elif sobolev_space == HCurl:
            gradient = appctx.get("discrete_gradient", None)
            if gradient is None:
                from firedrake.preconditioners.fdm import tabulate_exterior_derivative
                from firedrake.preconditioners.hiptmair import curl_to_grad
                Q = FunctionSpace(V.mesh(), curl_to_grad(V.ufl_element()))
                gradient = tabulate_exterior_derivative(Q, V)
                vertices = get_vertex_dofs(Q)

            bddcpc.setBDDCDiscreteGradient(gradient)

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


def get_vertex_dofs(V):
    W = FunctionSpace(V.mesh(), restrict(V.ufl_element(), "vertex"))
    indices = restricted_local_dofs(V, W)
    V.dof_dset.lgmap.apply(indices, result=indices)
    vertex_dofs = PETSc.IS().createGeneral(indices, comm=V.comm)
    return vertex_dofs
